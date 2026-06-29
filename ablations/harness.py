import math
import sys
from pathlib import Path

# Make the src package importable when run from the repo root, the same trick the
# entry scripts use: src/BEvAn is what lands on sys.path, so the
# top-level pipeline packages (physics, pipeline, modeling, ...) import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "BEvAn"))

import torch
from batch_analysis import discover_datasets

from modeling.matrix_calculations import bins_from_counts, build_density_matrix_1d, lookup_density_values_1d
from modeling.metrics import best_f1_threshold, metrics, roc_auc
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.datasets import BUCKETS, Datasets
from pipeline.eval import Evaluator, prior_free_scores
from pipeline.model_selection import apply_offset, calibrate_global_offset, search_hyperparams
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

# Floor inside the per-term log densities (mirrors modeling.calculate_probabilities.R).
EPS = 1e-8
# The three physics factors, in plot/column order; each maps to one pipeline term.
FACTORS = ("delta_E", "arm", "anni")
FACTOR_INDEX = {f: i for i, f in enumerate(FACTORS)}
# Per-factor 1D-histogram spacing (matching how the pipeline bins each feature):
# delta_E is log-spaced with a small floor; arm is now a signed score in [-1, 1]
# (511-aware ARM), so it is linear-binned to match the pipeline; anni is linear.
FACTOR_SPACING = {"delta_E": ("log", 1e-3), "arm": ("linear", None), "anni": ("linear", None)}
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
    "best_f1_record",
    "best_f1_threshold",
    "bucket_full_scores",
    "champion",
    "discover_datasets",
    "extract_split",
    "factor_1d_llr",
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


def extract_pool(ds: dict[str, str], ordering: str = "ckd", cache_dir: Path = CACHE_DIR) -> dict:
    """Extract (and cache) one dataset's flat, relabel-able event pool.

    Streams the ``.sim`` exactly once into the label-free pool (per-event features,
    feature bucket, and the n_std-free label score) so the ``gt_tolerance`` sweep can
    relabel for every window in memory instead of re-extracting per value. The ``.tra``
    is still back-projected once for the ARM source direction. Cached to
    ``cache_dir/<name>_<ordering>_pool.pt``.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).
        cache_dir: Directory for the cached ``.pt`` pool.

    Returns:
        The flat per-event pool dict from :meth:`Datasets.compute_event_pool`.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ds['name']}_{ordering}_pool.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)["pool"]

    imager = FarFieldImager(geometry_file=ds["geo_file"], n_phi=360, n_theta=180, coordinate_system="spheric")
    imager.backproject_file(ds["tra_file"])
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(ds["geo_file"], ds["sim_file"])
    pool = Datasets(reader_sim, unit_vector, ordering=ordering).compute_event_pool()

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pool": pool}, cache_path)
    return pool


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


def pooled_counts(evaluator: Evaluator, data: dict) -> dict:
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
    counts = pooled_counts(evaluator, data)
    return {**counts, **metrics(counts), **prior_free_scores(evaluator, data)}


def best_f1_record(trainer: Trainer, train: dict, eval_data: dict) -> dict:
    """Full-model metric record at its best-F1 (train-calibrated) operating point.

    The fair reference for a baseline that *also* tunes its threshold for F1 (the learned
    weights): both then sit at their best-F1 cut on the same pooled population, so the
    comparison isolates model quality from the operating point — not the deployed
    count-prior point, which is not F1-optimized and would hand a F1-tuned baseline an
    unfair edge.

    Args:
        trainer: The fitted champion trainer.
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.

    Returns:
        The eval-split record (counts, precision/recall/F1, best_f1, AUC) at the
        train-calibrated best-F1 threshold on the pooled per-event log-ratio.
    """
    evaluator = Evaluator(trainer)
    train_pool = evaluator.pooled_log_ratio(train)
    eval_pool = evaluator.pooled_log_ratio(eval_data)
    if train_pool is None or eval_pool is None:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}
        return {**counts, **metrics(counts), "best_f1": float("nan"), "auc": float("nan")}

    llr_train, y_train = train_pool
    llr_eval, y_eval = eval_pool
    ft = torch.isfinite(llr_train)
    fe = torch.isfinite(llr_eval)
    llr_train, y_train = llr_train[ft], y_train[ft]
    llr_eval, y_eval = llr_eval[fe], y_eval[fe].bool()

    _, threshold = best_f1_threshold(llr_train, y_train)
    if math.isnan(threshold):  # no positives on train -> keep the default 0 cut
        threshold = 0.0
    predictions = llr_eval >= threshold
    counts = {
        "tp": int((predictions & y_eval).sum()),
        "fp": int((predictions & ~y_eval).sum()),
        "fn": int((~predictions & y_eval).sum()),
        "tn": int((~predictions & ~y_eval).sum()),
        "excluded": 0,
    }
    return {
        **counts,
        **metrics(counts),
        "best_f1": best_f1_threshold(llr_eval, y_eval)[0],
        "auc": roc_auc(llr_eval, y_eval),
    }


def term_factor(spec: dict) -> str:
    """Factor name a density term belongs to: its x-feature (1D) or y-feature (2D)."""
    return spec["xfeat"] if spec.get("kind") == "1d" else spec["yfeat"]


def all_factors_finite(bucket_features: dict) -> torch.Tensor:
    """Boolean mask of events finite across all three factor features in a bucket."""
    mask = torch.ones_like(bucket_features["delta_E"], dtype=torch.bool)
    for f in FACTORS:
        mask = mask & torch.isfinite(bucket_features[f])
    return mask


def factor_1d_llr(
    train: dict,
    eval_data: dict,
    factor: str,
    bucket: int = 3,
    rho_floor: float = 10.0,
    smoothing: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """1D-density log-ratio for one factor on the common all-features-finite population.

    Fits a single-feature density (signal vs background) for ``factor`` on the
    training events of ``bucket`` and scores the eval events of ``bucket``, both
    restricted to events finite in every factor — so all three factors are scored on
    the *same* population (bucket 3 = the >=3-hit events where the back-to-back
    hypothesis exists). This removes the per-active-bucket / double-count confound of
    the earlier per-term design.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.
        factor: One of :data:`FACTORS`.
        bucket: Hit-multiplicity bucket holding the common population (default 3).
        rho_floor: Target events per histogram cell for the 1D density.
        smoothing: Laplace pseudo-counts for the 1D density.

    Returns:
        ``(llr, labels)`` over the eval-bucket events, or None if a class is empty.
    """
    spacing, floor = FACTOR_SPACING[factor]
    tr_bd, tr_bg = train["bdecay"][bucket], train["bg"][bucket]
    ev_bd, ev_bg = eval_data["bdecay"][bucket], eval_data["bg"][bucket]

    fb = tr_bd[factor][all_factors_finite(tr_bd)]
    fg = tr_bg[factor][all_factors_finite(tr_bg)]
    eb = ev_bd[factor][all_factors_finite(ev_bd)]
    eg = ev_bg[factor][all_factors_finite(ev_bg)]
    if min(fb.numel(), fg.numel(), eb.numel(), eg.numel()) == 0:
        return None

    n_bins = bins_from_counts(min(fb.numel(), fg.numel()), d=1, rho_floor=rho_floor, min_bins=5)
    _, x_bins = build_density_matrix_1d(
        torch.cat([fb, fg]), n_bins_x=n_bins, spacing_x=spacing, log_x_floor=floor
    )
    beta, _ = build_density_matrix_1d(fb, x_bins=x_bins, smoothing=smoothing)
    bgm, _ = build_density_matrix_1d(fg, x_bins=x_bins, smoothing=smoothing)

    values = torch.cat([eb, eg])
    labels = torch.cat([torch.ones(eb.numel(), dtype=torch.bool), torch.zeros(eg.numel(), dtype=torch.bool)])
    p_beta = lookup_density_values_1d(values, beta, x_bins)
    p_bg = lookup_density_values_1d(values, bgm, x_bins)
    return torch.log(p_beta + EPS) - torch.log(p_bg + EPS), labels


def bucket_full_scores(trainer: Trainer, data: dict, bucket: int = 3) -> dict[str, float]:
    """Best-F1 and AUC of the full model's pooled log-ratio on one bucket's events.

    The reference our-model performance on the same population the factor bars use,
    so the dashed line in the factor-contributions figure is comparable to them.

    Args:
        trainer: The fitted champion trainer.
        data: The split to score.
        bucket: Hit-multiplicity bucket (default 3, the >=3-hit population).

    Returns:
        ``{"f1": ..., "auc": ...}`` (NaN if the bucket is empty).
    """
    out = Evaluator(trainer).bucket_log_ratio(data, bucket, trainer.models[bucket])
    if out is None:
        return {"f1": float("nan"), "auc": float("nan")}
    llr, ground_truths = out
    return {"f1": best_f1_threshold(llr, ground_truths)[0], "auc": roc_auc(llr, ground_truths)}


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
