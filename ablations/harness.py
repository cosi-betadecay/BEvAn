import sys
from pathlib import Path

# Make the src package importable when run from the repo root, the same trick the
# entry scripts use: src/betadecay-analysis is what lands on sys.path, so the
# top-level pipeline packages (dataset, physics, pipeline, ...) import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "betadecay-analysis"))

import torch
from batch_analysis import discover_datasets

from dataset.datasets import BUCKETS, Datasets
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator, best_f1_threshold, metrics, prior_free_scores, roc_auc
from pipeline.model_selection import apply_offset, calibrate_global_offset, search_hyperparams
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

# Floor inside the per-term log densities (mirrors modeling.calculate_probablities.R).
EPS = 1e-8
# The three physics factors, in plot/column order; each maps to one pipeline term.
FACTORS = ("delta_E", "arm", "anni")
FACTOR_INDEX = {f: i for i, f in enumerate(FACTORS)}
# Feature-cache directory: ablations/cache.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Pipeline symbols re-exported for the ablation modules and the driver, alongside
# the helpers defined below (listed so the re-exports do not read as unused).
__all__ = [
    "BUCKETS",
    "CACHE_DIR",
    "EPS",
    "FACTORS",
    "Evaluator",
    "Trainer",
    "best_f1_threshold",
    "champion",
    "discover_datasets",
    "extract_split",
    "factor_term_scores",
    "fit_logistic",
    "metric_record",
    "metrics",
    "prior_free_scores",
    "roc_auc",
    "term_factor",
    "term_llr_columns",
]


def extract_split(
    ds: dict[str, str], ordering: str = "ckd", cache_dir: Path = CACHE_DIR
) -> tuple[dict, dict]:
    """Extract (and cache) one dataset's train/eval feature split.

    Back-projects the ``.tra`` for the source direction, streams the ``.sim`` through
    the feature builder at the requested subset ``ordering``, and splits. The result
    is cached to ``cache_dir/<name>_<ordering>.pt`` so repeat calls (and every
    model-level ablation) skip the MEGAlib-bound extraction.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).
        cache_dir: Directory for the cached ``.pt`` feature splits.

    Returns:
        ``(train, eval)`` nested per-class, per-bucket feature dicts.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ds['name']}_{ordering}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["train"], payload["eval"]

    imager = FarFieldImager(geometry_file=ds["geo_file"], n_phi=360, n_theta=180, coordinate_system="spheric")
    imager.backproject_file(ds["tra_file"])
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(ds["geo_file"], ds["sim_file"])
    datasets = Datasets(reader_sim, unit_vector, ordering=ordering)
    train, eval_data = datasets.split_dataset(datasets.compute_event_features())

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train, "eval": eval_data}, cache_path)
    return train, eval_data


def champion(train: dict, n_iter: int = 50, calibrate: bool = True) -> tuple[Trainer, dict]:
    """Fit the champion model: champion-seeded search, then optional calibration.

    Reproduces the deployed pipeline (``search_hyperparams`` -> ``Trainer.fit`` ->
    guarded ``calibrate_global_offset``), so the returned trainer is the reference
    the ablations are compared against and the config is reused by the model-level
    ablations (so only the named component differs between rows).

    Args:
        train: Nested per-class, per-bucket training feature dict.
        n_iter: Champion-seeded search candidates (0 keeps the champion seed).
        calibrate: Whether to calibrate and deploy the F1 threshold offset.

    Returns:
        ``(trainer, config)`` — the fitted (optionally calibrated) trainer and the
        chosen ``Trainer`` config.
    """
    config, _ = search_hyperparams(train, n_iter=n_iter)
    trainer = Trainer(**config).fit(train)
    if calibrate:
        offset, _ = calibrate_global_offset(trainer, train, config)
        apply_offset(trainer, offset)
    return trainer, config


def _pooled_counts(evaluator: Evaluator, data: dict) -> dict:
    """Sum the per-bucket confusion counts for ``data`` (no logging side effects)."""
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}
    for b in BUCKETS:
        counts = evaluator.evaluate_bucket(data, b, evaluator.trainer.models[b])
        if counts is None:
            continue
        for key in total:
            total[key] += counts[key]
    return total


def metric_record(trainer: Trainer, data: dict) -> dict:
    """Confusion counts plus precision/recall/FPR/F1 and prior-free best_f1/AUC.

    The standard per-split record every comparison ablation returns; computed
    quietly (no W&B/console logging) so ablation runs do not collide on log keys.

    Args:
        trainer: A fitted trainer (its deployed decision rule is scored).
        data: The split to evaluate (e.g. the eval split).

    Returns:
        Merged ``{counts, metrics, prior_free_scores}`` dict for the split.
    """
    evaluator = Evaluator(trainer)
    counts = _pooled_counts(evaluator, data)
    return {**counts, **metrics(counts), **prior_free_scores(evaluator, data)}


def term_factor(spec: dict) -> str:
    """Factor name a density term belongs to: its x-feature (1D) or y-feature (2D)."""
    return spec["xfeat"] if spec.get("kind") == "1d" else spec["yfeat"]


def factor_term_scores(evaluator: Evaluator, data: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-factor ``(llr_scores, labels)`` from each factor's own pipeline term.

    Keeps the pipeline statistics as-is (the delta_E double-count inside the ARM and
    anni joints is retained), so each factor is scored on the events of the buckets
    where its term is active.

    Args:
        evaluator: Evaluator bound to the fitted champion trainer.
        data: The split to score.

    Returns:
        ``{factor: (scores, labels)}`` over the factors whose term is present.
    """
    collected: dict[str, tuple[list, list]] = {f: ([], []) for f in FACTORS}
    for b in BUCKETS:
        model = evaluator.trainer.models[b]
        prepared = evaluator.prepare_terms(data, b, model)
        if prepared is None:
            continue
        terms, ground_truths, _ = prepared
        for spec, (p_beta, p_bg) in zip(model["terms"], terms, strict=True):
            llr = torch.log(p_beta + EPS) - torch.log(p_bg + EPS)
            scores, labels = collected[term_factor(spec)]
            scores.append(llr)
            labels.append(ground_truths)
    return {
        factor: (torch.cat(scores), torch.cat(labels))
        for factor, (scores, labels) in collected.items()
        if scores
    }


def term_llr_columns(evaluator: Evaluator, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Design matrix of per-term LLRs and labels, pooled across buckets.

    Each event becomes a row of length ``len(FACTORS)`` holding its term LLRs, with
    a zero in any column whose term is not present in that event's bucket — so the
    columns are the three factors and missing terms contribute nothing.

    Args:
        evaluator: Evaluator bound to the fitted champion trainer.
        data: The split to build the matrix for.

    Returns:
        ``(X, y)`` with ``X`` shape ``(n_events, len(FACTORS))`` and boolean ``y``.
    """
    rows, labels = [], []
    for b in BUCKETS:
        model = evaluator.trainer.models[b]
        prepared = evaluator.prepare_terms(data, b, model)
        if prepared is None:
            continue
        terms, ground_truths, _ = prepared
        row = torch.zeros(ground_truths.shape[0], len(FACTORS))
        for spec, (p_beta, p_bg) in zip(model["terms"], terms, strict=True):
            row[:, FACTOR_INDEX[term_factor(spec)]] = torch.log(p_beta + EPS) - torch.log(p_bg + EPS)
        rows.append(row)
        labels.append(ground_truths)
    if not rows:
        return torch.empty(0, len(FACTORS)), torch.empty(0, dtype=torch.bool)
    return torch.cat(rows), torch.cat(labels)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, max_iter: int = 100) -> tuple[torch.Tensor, float]:
    """Fit a logistic regression ``sigmoid(x @ w + b)`` by full-batch L-BFGS.

    The MLE weighting of the per-term LLRs — used by the learned-weights ablation to
    show the optimum lands near equal weights. Deterministic (zero init, no sampling).

    Args:
        x: ``(n, d)`` design matrix of per-term LLRs.
        y: ``(n,)`` boolean labels.
        max_iter: L-BFGS iterations.

    Returns:
        ``(weights, bias)`` — the ``(d,)`` weight tensor and scalar bias.
    """
    x64, target = x.double(), y.double()
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(x64 @ weights + bias, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach(), float(bias.detach())
