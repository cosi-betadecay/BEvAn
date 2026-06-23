import math

import ROOT as M
import torch
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.physics_factors import annihilation_angle, arm, delta_E
from tqdm import tqdm
from utils.megalib_types import MFileEventsSim

# Hit-multiplicity buckets, keyed by which features are physically available.
# An event is routed by what event processing actually produced (feature
# availability), NOT raw GetNHTs(), so it always lands in a bucket whose model
# uses only features the event possesses:
#   1 -> delta_E only            (no >=2-hit subset)
#   2 -> delta_E + ARM           (>=2-hit subset, no usable back-to-back)
#   3 -> delta_E + ARM + anni    (usable back-to-back hypothesis)
BUCKETS = (1, 2, 3)
FEATURES = ("delta_E", "arm", "anni")
CLASSES = ("bdecay", "bg")


class Datasets:
    def __init__(
        self,
        reader_sim: MFileEventsSim,
        reconstructed_unit_vector: torch.Tensor,
    ) -> None:
        self.reader_sim = reader_sim
        self.reconstructed_unit_vector = reconstructed_unit_vector
        self.train_percentage = 0.8

    def feature_bucket(self, d_arm: float, d_anni: float) -> int:
        if math.isfinite(d_anni):
            return 3
        if math.isfinite(d_arm):
            return 2
        return 1

    def compute_event_features(self) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
        # data[class][bucket][feature] -> list of per-event floats. The class and
        # bucket are decided together, per event, so the label and the bucket
        # routing can never desync (they are co-located by construction).
        data = {cls: {b: {f: [] for f in FEATURES} for b in BUCKETS} for cls in CLASSES}

        for event_sim in tqdm(
            iter(lambda: self.reader_sim.GetNextEvent(), None),
            desc="Processing events",
            unit=" events",
        ):
            M.SetOwnership(event_sim, True)
            n_hits = event_sim.GetNHTs()

            if n_hits == 0:
                continue

            cls = "bdecay" if ground_truth_bdecay(event_sim) else "bg"

            energies, positions, sizes = event_data_processing(event_sim)
            if energies is None or positions is None or sizes is None:
                # Invalid processing -> no usable feature; park in bucket 1 with
                # NaN so it counts toward the class prior but is excluded from the
                # confusion matrix (NaN ratio) downstream.
                for f in FEATURES:
                    data[cls][1][f].append(float("nan"))
                continue

            _delta_E = float(delta_E(energies, sizes=sizes))
            _anni = float(annihilation_angle(positions, sizes=sizes, energies=energies))
            _arm = float(arm(energies, positions, self.reconstructed_unit_vector, sizes=sizes))

            b = self.feature_bucket(_arm, _anni)
            data[cls][b]["delta_E"].append(_delta_E)
            data[cls][b]["arm"].append(_arm)
            data[cls][b]["anni"].append(_anni)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Lists -> float32 tensors, same nested shape.
        return {
            cls: {
                b: {f: torch.tensor(data[cls][b][f], dtype=torch.float32) for f in FEATURES} for b in BUCKETS
            }
            for cls in CLASSES
        }

    def split_dataset(self, data: dict[str, dict[int, dict[str, torch.Tensor]]]) -> tuple[dict, dict]:
        generator = torch.Generator().manual_seed(42)
        train = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
        evaluate = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}

        for cls in CLASSES:
            for b in BUCKETS:
                n = data[cls][b]["delta_E"].shape[0]
                perm = torch.randperm(n, generator=generator)
                n_train = int(n * self.train_percentage)
                train_idx, eval_idx = perm[:n_train], perm[n_train:]
                for f in FEATURES:
                    train[cls][b][f] = data[cls][b][f][train_idx]
                    evaluate[cls][b][f] = data[cls][b][f][eval_idx]

        return train, evaluate
