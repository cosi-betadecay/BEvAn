import math
from typing import Any

import omegaconf
import ROOT as M
import torch
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.physics_factors import annihilation_angle, arm, compton_phi, delta_E
from tqdm import tqdm

# Buckets keyed by hit-multiplicity / feature availability:
#   1 -> delta_E only            (1-hit: no Compton/geometry handle)
#   2 -> delta_E + phi           (2-hit: Compton scatter angle from energies)
#   3 -> delta_E + ARM AND delta_E + anni   (>=3-hit: ONE stratum, both matrices
#                                            multiply into R; anni is a per-event
#                                            factor — dropped when it's NaN, not a
#                                            separate bucket)
BUCKETS = (1, 2, 3)
_FEATURES = ("delta_E", "arm", "anni", "phi")
_CLASSES = ("bdecay", "bg")


def _feature_bucket(d_arm: float, d_phi: float) -> int:
    """Route by the most-informative available geometry: >=3-hit (ARM finite)
    => 3 (both ARM and anni matrices apply there; anni handled per-event as an
    optional factor); 2-hit Compton angle => 2; else delta_E only => 1.
    """
    if math.isfinite(d_arm):
        return 3
    if math.isfinite(d_phi):
        return 2
    return 1


class Datasets:
    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
        ref_energy: float,
        reader_sim: Any,
        reader_tra: Any,
        reconstructed_unit_vector: torch.Tensor,
    ):
        self.cfg = cfg
        self.ref_energy = ref_energy
        self.reader_sim = reader_sim
        self.reader_tra = reader_tra
        self.reconstructed_unit_vector = reconstructed_unit_vector
        self.train_percentage = 0.8
        self.eval_percentage = 0.2

    def compute_event_features(self, cfg, ref_energy, reader_sim, reader_tra, reconstructed_unit_vector):
        # data[class][bucket][feature] -> list of per-event floats. The class and
        # bucket are decided together, per event, so the label and the bucket
        # routing can never desync (they are co-located by construction).
        data = {cls: {b: {f: [] for f in _FEATURES} for b in BUCKETS} for cls in _CLASSES}

        for event_sim in tqdm(
            iter(lambda: reader_sim.GetNextEvent(), None),
            desc="Processing events",
            unit=" events",
        ):
            M.SetOwnership(event_sim, True)
            n_hits = event_sim.GetNHTs()

            if n_hits == 0:
                continue

            cls = "bdecay" if ground_truth_bdecay(event_sim, ref_energy) else "bg"

            energies, positions, sizes = event_data_processing(event_sim)
            if energies is None or positions is None or sizes is None:
                # Invalid processing -> no usable feature; park in bucket 1 with
                # NaN so it counts toward the class prior but is excluded from the
                # confusion matrix (NaN ratio) downstream.
                for f in _FEATURES:
                    data[cls][1][f].append(float("nan"))
                continue

            _delta_E = float(delta_E(energies, sizes=sizes))
            _anni = float(annihilation_angle(positions, n_hits=n_hits, sizes=sizes, energies=energies))
            _arm = float(arm(energies, positions, cfg, reconstructed_unit_vector, sizes=sizes))
            _phi = float(compton_phi(energies, sizes=sizes))

            b = _feature_bucket(_arm, _phi)
            data[cls][b]["delta_E"].append(_delta_E)
            data[cls][b]["arm"].append(_arm)
            data[cls][b]["anni"].append(_anni)
            data[cls][b]["phi"].append(_phi)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Lists -> float32 tensors, same nested shape.
        return {
            cls: {
                b: {f: torch.tensor(data[cls][b][f], dtype=torch.float32) for f in _FEATURES}
                for b in BUCKETS
            }
            for cls in _CLASSES
        }

    def split_dataset(self, data):
        """Split 80/20 within every (class, bucket) group independently, so each
        bucket's model trains and evaluates on its own population."""
        generator = torch.Generator().manual_seed(42)
        train = {cls: {b: {} for b in BUCKETS} for cls in _CLASSES}
        evaluate = {cls: {b: {} for b in BUCKETS} for cls in _CLASSES}

        for cls in _CLASSES:
            for b in BUCKETS:
                n = data[cls][b]["delta_E"].shape[0]
                perm = torch.randperm(n, generator=generator)
                n_train = int(n * self.train_percentage)
                train_idx, eval_idx = perm[:n_train], perm[n_train:]
                for f in _FEATURES:
                    train[cls][b][f] = data[cls][b][f][train_idx]
                    evaluate[cls][b][f] = data[cls][b][f][eval_idx]

        return train, evaluate
