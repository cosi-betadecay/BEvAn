import warnings
from collections.abc import Sequence
from pathlib import Path

import ROOT as M
import torch
from tqdm import tqdm

from physics.ground_truths import bdecay_label_score, calculate_tolerance, ground_truth_bdecay
from pipeline.datasets import BUCKETS, CLASSES
from pipeline.train import Trainer
from utils.reader_extraction import load_geometry, open_sim_reader


def validate_prior_sims(sim_files: Sequence[str]) -> None:
    """Validate the prior ``.sim`` paths the same way the entry scripts validate files.

    Args:
        sim_files: Prior ``.sim`` paths, one per prior dataset.

    Raises:
        ValueError: If no file is given or a path is not a ``.sim`` file.
        FileNotFoundError: If a file does not exist.
    """
    if not sim_files:
        raise ValueError("At least one prior .sim file (--prior-sim) is required to build the class priors.")
    for sim_file in sim_files:
        sim_path = Path(sim_file)
        if sim_path.suffix != ".sim":
            raise ValueError(f"--prior-sim must be a .sim file, got: {sim_file}")
        if not sim_path.is_file():
            raise FileNotFoundError(f"Prior simulation file not found: {sim_file}")


def count_prior_events(geo_file: str, sim_files: Sequence[str]) -> dict[int, dict[str, int]]:
    """Count β⁺/bg events per hit-multiplicity bucket across the prior ``.sim`` files.

    Streams every event, applies the same label and inclusion rule as training
    (:func:`physics.ground_truths.ground_truth_bdecay`; zero-hit events skipped),
    and splits by raw hit multiplicity — ``min(n_hits, 3)``: 1 hit -> bucket 1,
    2 -> 2, 3+ -> 3. Nothing else is computed. Paths are assumed already validated — the entry scripts run
    :func:`validate_prior_sims` upfront and :func:`validate_prior_counts` on the
    result; this function raises nothing of its own.

    Args:
        geo_file: MEGAlib geometry setup file the ``.sim`` files were simulated with.
        sim_files: Prior ``.sim`` paths, one per prior dataset.

    Returns:
        ``counts[bucket][class]`` summed over all files.
    """
    counts = {b: dict.fromkeys(CLASSES, 0) for b in BUCKETS}
    # One geometry scan shared by every prior file; each reader is closed even
    # if its stream raises mid-file.
    geometry = load_geometry(geo_file)
    for sim_file in sim_files:
        reader_sim = open_sim_reader(geometry, sim_file)
        try:
            for event_sim in tqdm(
                iter(lambda: reader_sim.GetNextEvent(), None),  # noqa: B023
                desc=f"Counting priors in {Path(sim_file).stem}",
                unit=" events",
            ):
                M.SetOwnership(event_sim, True)
                n_hits = event_sim.GetNHTs()
                if n_hits == 0:
                    continue
                cls = "bdecay" if ground_truth_bdecay(event_sim) else "bg"
                counts[min(n_hits, 3)][cls] += 1
        finally:
            reader_sim.Close()
    return counts


def count_prior_pool(geo_file: str, sim_files: Sequence[str]) -> dict[str, torch.Tensor]:
    """Stream the prior ``.sim`` files once into a flat, window-free pool.

    The relabel-able counterpart of :func:`count_prior_events`: instead of applying
    the n_std-dependent label it stores each event's raw label score
    (:func:`physics.ground_truths.bdecay_label_score`) next to its hit-multiplicity
    bucket, so :func:`pool_prior_counts` can produce the counts for any window
    without re-streaming — the basis of the ``label_window`` sweep's per-window
    prior. Same inclusion rule and bucketing as :func:`count_prior_events`
    (zero-hit events skipped, bucket ``min(n_hits, 3)``). Paths are assumed already
    validated; this function raises nothing of its own.

    Args:
        geo_file: MEGAlib geometry setup file the ``.sim`` files were simulated with.
        sim_files: Prior ``.sim`` paths, one per prior dataset.

    Returns:
        Flat pool with ``bucket`` (long) and ``label_score`` (float64) tensors,
        one entry per non-empty event across all files.
    """
    buckets: list[int] = []
    scores: list[float] = []
    # One geometry scan shared by every prior file; each reader is closed even
    # if its stream raises mid-file.
    geometry = load_geometry(geo_file)
    for sim_file in sim_files:
        reader_sim = open_sim_reader(geometry, sim_file)
        try:
            for event_sim in tqdm(
                iter(lambda: reader_sim.GetNextEvent(), None),  # noqa: B023
                desc=f"Pooling priors in {Path(sim_file).stem}",
                unit=" events",
            ):
                M.SetOwnership(event_sim, True)
                n_hits = event_sim.GetNHTs()
                if n_hits == 0:
                    continue
                buckets.append(min(n_hits, 3))
                scores.append(bdecay_label_score(event_sim))
        finally:
            reader_sim.Close()
    return {
        "bucket": torch.tensor(buckets, dtype=torch.long),
        "label_score": torch.tensor(scores, dtype=torch.float64),
    }


def pool_prior_counts(pool: dict[str, torch.Tensor], n_std: int = 5) -> dict[int, dict[str, int]]:
    """Per-bucket β⁺/bg counts a prior pool implies at one photopeak window.

    Thresholds each pooled label score at ``calculate_tolerance(n_std)`` with the
    same strict ``<`` :func:`physics.ground_truths.ground_truth_bdecay` uses, so the
    result is identical to a fresh :func:`count_prior_events` over the same files at
    that window — but no ``.sim`` is re-read.

    Args:
        pool: Flat prior pool from :func:`count_prior_pool`.
        n_std: Resolution-sigma window for the label (default ``5``, the deployed one).

    Returns:
        ``counts[bucket][class]``, in the layout :func:`count_prior_events` returns.
    """
    is_beta = pool["label_score"] < calculate_tolerance(n_std)
    masks = {"bdecay": is_beta, "bg": ~is_beta}
    return {
        b: {cls: int((mask & (pool["bucket"] == b)).sum()) for cls, mask in masks.items()} for b in BUCKETS
    }


def validate_prior_counts(counts: dict[int, dict[str, int]]) -> None:
    """Validate counted priors — called by the entry scripts right after counting.

    Args:
        counts: ``counts[bucket][class]`` from :func:`count_prior_events`.

    Raises:
        ValueError: If any bucket has zero events in both classes (its prior,
            and hence its posterior, would be undefined).
    """
    for b in BUCKETS:
        n_beta, n_bg = counts[b]["bdecay"], counts[b]["bg"]
        if n_beta + n_bg == 0:
            raise ValueError(f"Prior files contain no events for bucket {b}; cannot form its prior.")
        if n_beta == 0 or n_bg == 0:
            warnings.warn(
                f"Bucket {b} prior has an empty class (n_beta={n_beta}, n_bg={n_bg}); "
                "its decisions become single-class.",
                stacklevel=2,
            )


def bucket_priors(counts: dict[int, dict[str, int]]) -> dict[int, float]:
    """The prior probability P(β⁺) each bucket's counts imply (NaN for an empty bucket).

    Args:
        counts: ``counts[bucket][class]`` from :func:`count_prior_events`.

    Returns:
        ``{bucket: P(β⁺)}`` for reporting/logging.
    """
    priors = {}
    for b in BUCKETS:
        n_beta, n_bg = counts[b]["bdecay"], counts[b]["bg"]
        total = n_beta + n_bg
        priors[b] = n_beta / total if total > 0 else float("nan")
    return priors


def apply_prior_counts(trainer: Trainer, counts: dict[int, dict[str, int]]) -> None:
    """Overwrite a fitted trainer's per-bucket prior counts with the external ones (in place).

    Only ``n_beta``/``n_bg`` change; the fitted density terms are untouched. The
    deployed cut is the count-prior posterior rule ``R >= n_bg / n_beta``, so
    these counts set each bucket's operating point directly.

    Buckets without fitted density terms are left untouched: keeping the training
    counts preserves each terms-less bucket's pre-change behavior (in the
    empty-training-class case the bucket stays degenerate; in the rarer
    both-classes-present case — every value of one class's features is NaN, so no
    density could be fit — the bucket keeps the same constant decision as before).

    Counts are assumed already validated (:func:`validate_prior_counts` in the
    entry scripts); this function raises nothing of its own.

    Args:
        trainer: A fitted trainer whose bucket models receive the prior.
        counts: ``counts[bucket][class]`` from :func:`count_prior_events`.
    """
    for b in BUCKETS:
        model = trainer.models[b]
        if not model["terms"]:
            continue
        model["n_beta"] = counts[b]["bdecay"]
        model["n_bg"] = counts[b]["bg"]
