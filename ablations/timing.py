import platform
import tempfile
import time
from pathlib import Path
from statistics import median

import torch

# _extract_payload is the private, uncached extraction, imported deliberately: every
# public extract_* wrapper in harness goes through the feature cache, which is exactly
# what must not be timed here.
from harness import DatasetPaths, Evaluator, Trainer, _extract_payload

from pipeline.datasets import BUCKETS, CLASSES, DEFAULT_SPLIT_SEED, split_features

# Repeats for the sub-second stages (fit, scoring), reported as the median.
REPEATS = 7
# Threads torch is pinned to, so a rerun on the same machine reproduces the number
# instead of tracking whatever else the machine happened to be doing.
THREADS = 1


def environment() -> dict[str, str | int | bool]:
    """Describe the machine and library versions a timing record was produced on.

    Returns:
        Platform, processor, Python and torch versions, torch's thread count, and
            whether CUDA was available (the pipeline is CPU-bound either way).
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def n_events(data: dict) -> int:
    """Count events in a nested per-class, per-bucket feature dict.

    Args:
        data: Nested ``{class: {bucket: {feature: tensor}}}`` feature dict.

    Returns:
        Total events across both classes and every bucket. ``delta_E`` is the
            counting feature because it is the one every bucket defines.
    """
    return sum(int(data[c][b]["delta_E"].numel()) for c in CLASSES for b in BUCKETS)


def n_density_cells(trainer: Trainer) -> int:
    """Count the fitted density cells a scoring lookup indexes into.

    Args:
        trainer: A fitted trainer.

    Returns:
        Total cells over every bucket's terms, both classes counted — the size of
            the table the deployed decision consults per event.
    """
    return sum(
        int(term[cls].numel())
        for model in trainer.models.values()
        for term in model["terms"]
        for cls in ("beta", "bg")
    )


def artifact_kb(trainer: Trainer) -> float:
    """Measure the on-disk size of a fitted trainer's saved artifact.

    Written to a temporary directory and discarded: the number wanted is the size of
    what deployment would ship, not a file kept around.

    Args:
        trainer: A fitted trainer.

    Returns:
        Size of the ``.pt`` artifact in kB.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        trainer.save(path)
        return path.stat().st_size / 1024


def run(
    ds: DatasetPaths,
    config: dict | None = None,
    repeats: int = REPEATS,
    threads: int = THREADS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, float | int]:
    """Time extraction, fit and scoring for one dataset off a single uncached pass.

    The extraction is run once with the cache bypassed and its payload reused for the
    remaining stages, so no stage is measured against data another stage did not
    produce. Fit and scoring are repeated and reported as medians.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        config: ``Trainer`` config; ``None`` uses the reference budget, which is what
            :func:`harness.reference` deploys.
        repeats: Repeats for the fit and scoring stages.
        threads: Thread count torch is pinned to for the duration.
        seed: Split seed for the train/eval partition.

    Returns:
        Flat per-stage record: event counts, ``extract_s``/``fit_s``/``score_s``,
            the per-event extraction cost in µs, batch scoring throughput in events
            per second, the per-event extraction-to-scoring cost ratio, the saved
            artifact size in kB, and the density-cell count.
    """
    torch.set_num_threads(threads)

    start = time.perf_counter()
    payload = _extract_payload(ds, "ckd")
    extract_s = time.perf_counter() - start

    data = payload["data"]
    total_events = n_events(data)
    train, eval_data = split_features(data, seed=seed)
    eval_events = n_events(eval_data)

    fit_times = []
    for _ in range(repeats):
        trainer = Trainer(**(config or {}))
        start = time.perf_counter()
        trainer.fit(train)
        fit_times.append(time.perf_counter() - start)

    evaluator = Evaluator(trainer)
    evaluator.pooled_log_ratio(eval_data)  # warm-up, discarded
    score_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        evaluator.pooled_log_ratio(eval_data)
        score_times.append(time.perf_counter() - start)

    fit_s = median(fit_times)
    score_s = median(score_times)
    extract_per_event = extract_s / total_events if total_events else float("nan")
    score_per_event = score_s / eval_events if eval_events else float("nan")

    return {
        "n_events": total_events,
        "n_eval_events": eval_events,
        "extract_s": extract_s,
        "extract_us_per_event": 1e6 * extract_per_event,
        "fit_s": fit_s,
        "score_s": score_s,
        "score_events_per_s_batched": eval_events / score_s if score_s > 0 else float("nan"),
        "extract_over_score_per_event": extract_per_event / score_per_event
        if score_per_event
        else float("nan"),
        "artifact_kb": artifact_kb(trainer),
        "n_density_cells": n_density_cells(trainer),
    }
