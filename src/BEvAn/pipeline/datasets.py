import math

import ROOT as M
import torch
from tqdm import tqdm

from physics.event_processing import event_data_processing
from physics.ground_truths import bdecay_label_score, calculate_tolerance, ground_truth_bdecay
from physics.physics_factors import annihilation_angle, arm, delta_E
from utils.megalib_types import MFileEventsSim
from utils.reader_extraction import ChunkScopedIds

# Hit-multiplicity buckets, keyed by which features are physically available.
# An event is routed by what event processing actually produced (feature
# availability), not raw GetNHTs(), so it always lands in a bucket whose model
# uses only features the event possesses:
#   1 -> delta_E only            (no >=2-hit subset)
#   2 -> delta_E + ARM           (>=2-hit subset, no usable back-to-back)
#   3 -> delta_E + ARM + anni    (usable back-to-back hypothesis)
BUCKETS = (1, 2, 3)
FEATURES = ("delta_E", "arm", "anni")
CLASSES = ("bdecay", "bg")
# The pipeline's canonical split seed: every single-split entry point uses it, and
# the multi-seed uncertainty runs start from it (seeds DEFAULT_SPLIT_SEED + i) so
# their first seed reproduces the single-split numbers exactly.
DEFAULT_SPLIT_SEED = 42


class Datasets:
    """Build the per-event feature dataset for the β/bg classifier from a MEGAlib reader.

    Streams every event, labels it β⁺ decay or background via the ANNI ground
    truth, computes the three physics features (delta_E, ARM, annihilation
    angle) over the event's hit subsets, and routes each event to a
    hit-multiplicity bucket by which of those features it actually produced. Also
    splits the assembled features into reproducible train/eval partitions.
    """

    def __init__(
        self,
        reader_sim: MFileEventsSim,
        reconstructed_unit_vector: torch.Tensor,
        ordering: str = "ckd",
    ) -> None:
        """Bind the dataset builder to a simulation reader and source direction.

        Args:
            reader_sim: Open MEGAlib ``.sim`` event reader to stream events from.
            reconstructed_unit_vector: Reconstructed source direction the ARM
                feature is measured against.
            ordering: Hit-subset ordering passed to event processing — ``"ckd"``
                (the default Compton-kinematic-discrepancy ordering) or ``"energy"``
                (plain energy sort), the latter used by the no-CKD-order ablation.
        """
        self.reader_sim = reader_sim
        self.reconstructed_unit_vector = reconstructed_unit_vector
        self.ordering = ordering

    def feature_bucket(self, d_arm: float, d_anni: float) -> int:
        """Route an event to its feature bucket from which features are finite.

        Returns 3 if the back-to-back annihilation score exists, 2 if only ARM
        exists, else 1 (delta_E only).
        """
        if math.isfinite(d_anni):
            return 3
        if math.isfinite(d_arm):
            return 2
        return 1

    def compute_event_features(self) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
        """Stream every event from the reader into per-class, per-bucket feature tensors.

        Each event is labeled β⁺ decay/bg, expanded into hit-subset energies and
        positions, and reduced to its delta_E, ARM, and annihilation-angle
        features. Events with no usable processing are parked in bucket 1 with
        NaN so they still count toward the class prior but are excluded downstream.

        Returns:
            Nested dict ``data[class][bucket][feature]`` of per-event float tensors.
        """
        return self.compute_event_features_keyed()[0]

    def compute_event_features_keyed(
        self,
    ) -> tuple[dict[str, dict[int, dict[str, torch.Tensor]]], dict[str, dict[int, list[tuple[int, int]]]]]:
        """Stream events into feature tensors plus their chunk-scoped event keys.

        The feature side is exactly :meth:`compute_event_features` (that method is a
        thin wrapper over this one). Alongside, each event's chunk-scoped
        ``(chunk, id)`` key (see :class:`~utils.reader_extraction.ChunkScopedIds`) is
        appended to a parallel nesting in the same loop iteration, so
        ``keys[class][bucket][i]`` names the event behind row ``i`` of every
        ``data[class][bucket][feature]`` tensor — the co-location invariant extended
        to event identity. The keys join events to the same simulation's ``.tra``.

        Returns:
            ``(data, keys)`` — the nested ``data[class][bucket][feature]`` float
            tensors and the parallel ``keys[class][bucket]`` lists of event keys.
        """
        # data[class][bucket][feature] -> list of per-event floats. The class and
        # bucket are decided together, per event, so the label and the bucket
        # routing can never desync (they are co-located by construction).
        data = {cls: {b: {f: [] for f in FEATURES} for b in BUCKETS} for cls in CLASSES}
        keys = {cls: {b: [] for b in BUCKETS} for cls in CLASSES}
        ids = ChunkScopedIds()

        for event_sim in tqdm(
            iter(lambda: self.reader_sim.GetNextEvent(), None),
            desc="Processing events",
            unit=" events",
        ):
            M.SetOwnership(event_sim, True)
            # Every event must advance the chunk-scoped ID tracking, including the
            # empty ones skipped below — a skipped ID reset would desync the chunks.
            key = ids.key(int(event_sim.GetID()))
            n_hits = event_sim.GetNHTs()

            if n_hits == 0:
                continue

            cls = "bdecay" if ground_truth_bdecay(event_sim) else "bg"

            energies, positions, sizes = event_data_processing(event_sim, ordering=self.ordering)
            if energies is None or positions is None or sizes is None:
                # Invalid processing -> no usable feature; park in bucket 1 with
                # NaN so it counts toward the class prior but is excluded from the
                # confusion matrix (NaN ratio) downstream.
                for f in FEATURES:
                    data[cls][1][f].append(float("nan"))
                keys[cls][1].append(key)
                continue

            _delta_E = float(delta_E(energies, sizes=sizes))
            _anni = float(annihilation_angle(positions, sizes=sizes, energies=energies))
            _arm = float(arm(energies, positions, self.reconstructed_unit_vector, sizes=sizes))

            b = self.feature_bucket(_arm, _anni)
            data[cls][b]["delta_E"].append(_delta_E)
            data[cls][b]["arm"].append(_arm)
            data[cls][b]["anni"].append(_anni)
            keys[cls][b].append(key)

        # Lists -> float32 tensors, same nested shape.
        tensors = {
            cls: {
                b: {f: torch.tensor(data[cls][b][f], dtype=torch.float32) for f in FEATURES} for b in BUCKETS
            }
            for cls in CLASSES
        }
        return tensors, keys

    def compute_event_pool(self) -> dict[str, torch.Tensor]:
        """Stream every event once into a flat, relabel-able per-event pool.

        Mirrors :meth:`compute_event_features` event-for-event, but instead of applying
        the n_std-dependent label it stores each event's raw label score (see
        :func:`bdecay_label_score`) next to its features and feature bucket. Bucket and
        features do not depend on the label, so :func:`pool_to_features` can rebuild the
        labeled, bucketed dict for any window without re-streaming — the basis of the
        cheap ``label_window`` sweep. The score is taken before processing, matching the
        label-then-processing order in :meth:`compute_event_features` so a relabel at
        ``n_std = 5`` reproduces that method exactly.

        Returns:
            Flat pool with ``bucket`` (long), ``delta_E``/``arm``/``anni`` (float32), and
            ``label_score`` (float64) tensors, one entry per non-empty event.
        """
        buckets: list[int] = []
        feats: dict[str, list[float]] = {f: [] for f in FEATURES}
        scores: list[float] = []

        for event_sim in tqdm(
            iter(lambda: self.reader_sim.GetNextEvent(), None),
            desc="Pooling events",
            unit=" events",
        ):
            M.SetOwnership(event_sim, True)
            if event_sim.GetNHTs() == 0:
                continue

            # Score before processing, mirroring compute_event_features (which labels
            # before processing) so the relabeled pool matches it event-for-event.
            score = bdecay_label_score(event_sim)
            energies, positions, sizes = event_data_processing(event_sim, ordering=self.ordering)
            if energies is None or positions is None or sizes is None:
                # Invalid processing -> bucket 1 with NaN features (excluded downstream
                # by the NaN ratio) but still carrying its label score for the prior.
                buckets.append(1)
                for f in FEATURES:
                    feats[f].append(float("nan"))
                scores.append(score)
                continue

            _delta_E = float(delta_E(energies, sizes=sizes))
            _anni = float(annihilation_angle(positions, sizes=sizes, energies=energies))
            _arm = float(arm(energies, positions, self.reconstructed_unit_vector, sizes=sizes))

            buckets.append(self.feature_bucket(_arm, _anni))
            feats["delta_E"].append(_delta_E)
            feats["arm"].append(_arm)
            feats["anni"].append(_anni)
            scores.append(score)

        return {
            "bucket": torch.tensor(buckets, dtype=torch.long),
            **{f: torch.tensor(feats[f], dtype=torch.float32) for f in FEATURES},
            "label_score": torch.tensor(scores, dtype=torch.float64),
        }

    def split_dataset(
        self,
        data: dict[str, dict[int, dict[str, torch.Tensor]]],
        train_percentage: float = 0.8,
        seed: int = DEFAULT_SPLIT_SEED,
    ) -> tuple[dict[str, dict[int, dict[str, torch.Tensor]]], dict[str, dict[int, dict[str, torch.Tensor]]]]:
        """Split each per-class, per-bucket feature tensor into train/eval partitions.

        Thin instance-method wrapper over :func:`split_features` (the actual logic),
        kept so the split-using entry points (``analysis.py``, the ablation harness)
        call it unchanged; ``train_model.py`` fits on the full feature dict and never
        splits.

        Args:
            data: Nested feature dict from :meth:`compute_event_features`.
            train_percentage: Fraction of each class/bucket assigned to the train
                partition; the remainder is the eval partition.
            seed: Shuffle seed for the partition (default :data:`DEFAULT_SPLIT_SEED`).

        Returns:
            ``(train, evaluate)``, each mirroring the nested structure of ``data``.
        """
        return split_features(data, train_percentage, seed)


def pool_to_features(
    pool: dict[str, torch.Tensor], n_std: int
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """Relabel a flat event pool at one photopeak window into the nested feature dict.

    Thresholds each event's cached label score at ``calculate_tolerance(n_std)`` to
    assign β⁺/background, then groups by the (label-independent) feature bucket. The
    result is identical to a fresh :meth:`Datasets.compute_event_features` whose ground
    truth used ``n_std`` — same features, same routing, same per-bucket event order — so
    the downstream :func:`split_features` and fit are unchanged, but no ``.sim`` is
    re-read. This is what lets the ``label_window`` sweep cost one extraction, not one
    per window.

    Args:
        pool: Flat per-event pool from :meth:`Datasets.compute_event_pool`.
        n_std: Resolution-sigma window for the label.

    Returns:
        Nested ``data[class][bucket][feature]`` dict, ready for :func:`split_features`.
    """
    tolerance = calculate_tolerance(n_std)
    is_beta = pool["label_score"] < tolerance
    masks = {"bdecay": is_beta, "bg": ~is_beta}
    return {
        cls: {b: {f: pool[f][masks[cls] & (pool["bucket"] == b)] for f in FEATURES} for b in BUCKETS}
        for cls in CLASSES
    }


def split_features(
    data: dict[str, dict[int, dict[str, torch.Tensor]]],
    train_percentage: float = 0.8,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[dict[str, dict[int, dict[str, torch.Tensor]]], dict[str, dict[int, dict[str, torch.Tensor]]]]:
    """Split each per-class, per-bucket feature tensor into reproducible train/eval partitions.

    Seeded shuffle per ``(class, bucket)``, so the partition is deterministic given
    the class membership and ``seed`` — relabeling then splitting (the ``label_window``
    path) lands on exactly the partition a fresh extraction at that label would. Varying
    ``seed`` (the multi-seed uncertainty runs) is the only supported way to draw a
    different partition.

    Args:
        data: Nested per-class, per-bucket feature dict.
        train_percentage: Fraction of each class/bucket assigned to the train partition;
            the remainder is the eval partition.
        seed: Shuffle seed for the partition (default :data:`DEFAULT_SPLIT_SEED`).

    Returns:
        ``(train, evaluate)``, each mirroring the nested structure of ``data``.
    """
    generator = torch.Generator().manual_seed(seed)
    train = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
    evaluate = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}

    for cls in CLASSES:
        for b in BUCKETS:
            n = data[cls][b]["delta_E"].shape[0]
            perm = torch.randperm(n, generator=generator)
            n_train = int(n * train_percentage)
            train_idx, eval_idx = perm[:n_train], perm[n_train:]
            for f in FEATURES:
                train[cls][b][f] = data[cls][b][f][train_idx]
                evaluate[cls][b][f] = data[cls][b][f][eval_idx]

    return train, evaluate
