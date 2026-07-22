import math
import sys
from pathlib import Path
from typing import TypedDict

# Make the src package importable when run from the repo root, the same trick the
# entry scripts use: src/BEvAn is what lands on sys.path, so the
# top-level pipeline packages (physics, pipeline, modeling, ...) import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "BEvAn"))

import torch

from modeling.matrix_calculations import build_density_matrix_1d, lookup_density_values_1d
from modeling.metrics import best_f1_threshold, metrics, roc_auc
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.datasets import BUCKETS, DEFAULT_SPLIT_SEED, Datasets, split_features
from pipeline.eval import Evaluator, prior_free_scores
from pipeline.priors import apply_prior_counts, count_prior_pool, pool_prior_counts, validate_prior_counts
from pipeline.train import Trainer
from utils.local_results import save_confusion_matrix, save_density_terms, save_figure, save_score_curves
from utils.plots import plot_pr_curve, plot_roc_curve
from utils.reader_extraction import load_geometry, open_sim_reader

# Floor inside the per-term log densities (mirrors modeling.calculate_probabilities.log_R).
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
# Dataset folders live one per name under data/ at the repo root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Static dataset-name -> geometry setup map. Each dataset folder in data/ was
# produced against a different geometry, so discovery cannot take a single
# --geo-file flag; the per-dataset entry points (analysis.py, train_model.py,
# inference.py) pass geometry explicitly instead.
GEOMETRIES = {
    "SPILike": "$MEGALIB/resource/examples/geomega/special/SPILike.geo.setup",
    "NCT": "$MEGALIB/resource/examples/geomega/special/Simple_Ge_NCT_try1_040721.geo.setup",
    "Max": "$MEGALIB/resource/examples/geomega/special/Max.geo.setup",
    "GeACT": "$MEGALIB/resource/examples/geomega/special/GeACT.geo.setup",
    "COSIBalloon_9det": "$MEGALIB/resource/examples/geomega/cosiballoon/COSIBalloon.9Detector.geo.setup",
    "COSIBalloon_10det": "$MEGALIB/resource/examples/geomega/cosiballoon/COSIBalloon.10Detector.geo.setup",
    "COSIBalloon_12det": "$MEGALIB/resource/examples/geomega/cosiballoon/COSIBalloon.12Detector.geo.setup",
}

# Pipeline symbols re-exported for the ablation modules and the driver, alongside
# the helpers defined below (listed so the re-exports do not read as unused).
__all__ = [
    "BUCKETS",
    "CACHE_DIR",
    "DEFAULT_SPLIT_SEED",
    "EPS",
    "FACTORS",
    "Evaluator",
    "Trainer",
    "aggregate_seed_records",
    "best_f1_record",
    "best_f1_threshold",
    "bucket_full_scores",
    "dedicated_prior_counts",
    "deploy_prior",
    "discover_datasets",
    "extract_keyed",
    "extract_prior_pool",
    "extract_split",
    "factor_1d_llr",
    "fit_logistic",
    "metric_record",
    "metrics",
    "prior_free_scores",
    "prior_pool_counts",
    "reference",
    "roc_auc",
    "save_density_terms",
    "save_model_plots",
    "save_score_plots",
    "sky_imager",
    "term_factor",
    "term_llr_columns",
]


class DatasetPaths(TypedDict):
    """One discovered dataset folder's paths (built by :func:`discover_datasets`)."""

    name: str
    geo_file: str
    sim_file: str
    tra_file: str
    prior_sims: list[str]


def discover_datasets() -> list[DatasetPaths]:
    """Find every complete dataset folder ``data/<name>/`` with a registered geometry.

    A dataset folder holds ``<name>.sim`` + ``<name>.tra`` (the train/eval pair)
    and ``<name>_P1.sim`` / ``<name>_P2.sim`` — the dedicated prior simulations
    whose per-bucket event counts form the class priors (:mod:`pipeline.priors`),
    deployed by the ``dedicated_prior`` ablation. A folder is skipped (with a message)
    if the ``.tra``, the prior files, or the :data:`GEOMETRIES` entry is missing;
    folders with no ``.sim`` at all (e.g. ``data/cosima/``) are ignored silently.

    Returns:
        One :class:`DatasetPaths` per dataset: name, geo_file, sim_file, and
            tra_file (str) plus prior_sims (list[str]).
    """
    datasets: list[DatasetPaths] = []
    for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        name = folder.name
        sim_file = folder / f"{name}.sim"
        if not sim_file.is_file():
            if any(folder.glob("*.sim")):
                print(f"Skipping {name}: no {sim_file.name} in {folder}")
            continue
        tra_file = sim_file.with_suffix(".tra")
        if not tra_file.exists():
            print(f"Skipping {name}: no matching {tra_file.name}")
            continue
        geo_file = GEOMETRIES.get(name)
        if geo_file is None:
            print(f"Skipping {name}: no geometry registered in GEOMETRIES")
            continue
        prior_sims = sorted(str(p) for p in folder.glob(f"{name}_P*.sim"))
        if not prior_sims:
            print(f"Skipping {name}: no {name}_P*.sim prior files")
            continue
        datasets.append(
            {
                "name": name,
                "geo_file": geo_file,
                "sim_file": str(sim_file),
                "tra_file": str(tra_file),
                "prior_sims": prior_sims,
            }
        )
    return datasets


def sky_imager(ds: DatasetPaths) -> FarFieldImager:
    """Far-field imager on the dataset's geometry, at the standard sky binning.

    Args:
        ds: Dataset paths with a ``geo_file`` key.

    Returns:
        A fresh :class:`FarFieldImager` (360x180 spheric grid, the pipeline default).
    """
    return FarFieldImager(geometry_file=ds["geo_file"], n_phi=360, n_theta=180, coordinate_system="spheric")


def _extract_payload(ds: DatasetPaths, ordering: str) -> dict:
    """Run the full MEGAlib-bound extraction for one dataset.

    Back-projects the ``.tra`` (which yields both the ARM source direction and the
    all-events sky image) and streams the ``.sim`` through the keyed feature builder.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).

    Returns:
        Cache payload with ``data``/``keys`` (the keyed feature dicts) and
        ``sky_image``/``sky_n_events`` (the all-events backprojection).
    """
    imager = sky_imager(ds)
    n_events = imager.backproject_file(ds["tra_file"])
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = open_sim_reader(load_geometry(ds["geo_file"]), ds["sim_file"])
    data, keys = Datasets(reader_sim, unit_vector, ordering=ordering).compute_event_features_keyed()
    return {"data": data, "keys": keys, "sky_image": imager.image_2d(), "sky_n_events": n_events}


def _cached_payload(ds: DatasetPaths, ordering: str, cache_dir: Path, require_keys: bool = False) -> dict:
    """Load one dataset's extraction cache, extracting (and writing it) on a miss.

    Legacy caches predate the event keys and hold only the pre-split ``train``/``eval``
    dicts; they are served as-is unless ``require_keys`` forces the one-time
    re-extraction that upgrades the file to the keyed layout.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).
        cache_dir: Directory for the cached ``.pt`` payloads.
        require_keys: Whether the caller needs the keyed layout.

    Returns:
        The cache payload (keyed layout, or legacy when allowed).
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ds['name']}_{ordering}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not require_keys or "keys" in payload:
            return payload
        print(f"    {cache_path.name} predates event keys; re-extracting once to upgrade it")

    payload = _extract_payload(ds, ordering)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def extract_split(
    ds: DatasetPaths,
    ordering: str = "ckd",
    cache_dir: Path = CACHE_DIR,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[dict, dict]:
    """Extract (and cache) one dataset's train/eval feature split.

    Back-projects the ``.tra`` for the source direction, streams the ``.sim`` through
    the feature builder at the requested subset ``ordering``, and splits. The result
    is cached to ``cache_dir/<name>_<ordering>.pt`` so repeat calls (and every
    model-level ablation) skip the MEGAlib-bound extraction. The split is derived
    from the cached unsplit features by the pipeline's seeded
    :func:`~pipeline.datasets.split_features`, so it is identical run to run at a
    given ``seed`` — the multi-seed uncertainty loop only re-splits the same cached
    pool. Legacy caches that stored the split directly pin the default seed and are
    refused for any other.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).
        cache_dir: Directory for the cached ``.pt`` feature payloads.
        seed: Split seed (default :data:`~pipeline.datasets.DEFAULT_SPLIT_SEED`).

    Returns:
        ``(train, eval)`` nested per-class, per-bucket feature dicts.

    Raises:
        RuntimeError: If a legacy pre-split cache is asked for a non-default seed.
    """
    payload = _cached_payload(ds, ordering, cache_dir)
    if "data" in payload:
        return split_features(payload["data"], seed=seed)
    if seed != DEFAULT_SPLIT_SEED:
        raise RuntimeError(
            f"Cache for {ds['name']} ({ordering}) predates the unsplit layout and pins the "
            f"default split; delete it to re-extract before running with seed={seed}."
        )
    return payload["train"], payload["eval"]


def extract_keyed(
    ds: DatasetPaths, ordering: str = "ckd", cache_dir: Path = CACHE_DIR
) -> tuple[dict, dict, torch.Tensor, int]:
    """Extract (and cache) one dataset's keyed features plus the all-events sky image.

    The dedicated_prior sky figure needs event identities (to join classifier tags back
    to the ``.tra`` cones) and the all-events backprojection (its reference panel) —
    both produced by the same extraction :func:`extract_split` caches. A legacy
    (pre-key) cache triggers one re-extraction that upgrades the file in place;
    every other ablation keeps reading it either way.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        ordering: Subset ordering for the feature builder (``"ckd"`` or ``"energy"``).
        cache_dir: Directory for the cached ``.pt`` feature payloads.

    Returns:
        ``(data, keys, sky_image, sky_n_events)`` — the unsplit per-class,
        per-bucket feature dict, its parallel event-key lists, and the all-events
        ``(n_theta, n_phi)`` sky image with its accumulated event count.
    """
    payload = _cached_payload(ds, ordering, cache_dir, require_keys=True)
    return payload["data"], payload["keys"], payload["sky_image"], payload["sky_n_events"]


def extract_pool(ds: DatasetPaths, ordering: str = "ckd", cache_dir: Path = CACHE_DIR) -> dict:
    """Extract (and cache) one dataset's flat, relabel-able event pool.

    Streams the ``.sim`` exactly once into the label-free pool (per-event features,
    feature bucket, and the n_std-free label score) so the ``label_window`` sweep can
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

    imager = sky_imager(ds)
    imager.backproject_file(ds["tra_file"])
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = open_sim_reader(load_geometry(ds["geo_file"]), ds["sim_file"])
    pool = Datasets(reader_sim, unit_vector, ordering=ordering).compute_event_pool()

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pool": pool}, cache_path)
    return pool


def extract_prior_pool(ds: DatasetPaths, cache_dir: Path = CACHE_DIR) -> dict:
    """Extract (and cache) one dataset's relabel-able dedicated-prior pool.

    Streams the dedicated ``<name>_P*.sim`` files exactly once into the window-free
    pool of label scores + hit-multiplicity buckets, so the ``label_window`` sweep can
    re-derive the deployed class priors at every window
    (:func:`~pipeline.priors.pool_prior_counts`) instead of re-reading the prior files
    once per value. Cached to ``cache_dir/<name>_prior_pool.pt``.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``prior_sims`` keys.
        cache_dir: Directory for the cached ``.pt`` pool.

    Returns:
        The flat prior pool dict from :func:`~pipeline.priors.count_prior_pool`.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ds['name']}_prior_pool.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)["pool"]

    pool = count_prior_pool(ds["geo_file"], ds["prior_sims"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pool": pool}, cache_path)
    return pool


def prior_pool_counts(prior_pool: dict, n_std: int = 5) -> dict:
    """Validated dedicated-prior counts from an already-extracted prior pool.

    The single source of the deployed operating point for every prior-dependent
    ablation: counts derived from the dedicated prior files, never from a training
    split. Taking the pool (rather than the dataset) lets the ``label_window`` sweep
    re-derive the counts per window from one in-memory pool.

    Args:
        prior_pool: Flat prior pool from :func:`extract_prior_pool`.
        n_std: Resolution-sigma window for the label (default ``5``, the deployed one).

    Returns:
        ``counts[bucket][class]``, in the layout
        :func:`~pipeline.priors.count_prior_events` returns.
    """
    counts = pool_prior_counts(prior_pool, n_std)
    validate_prior_counts(counts)
    return counts


def dedicated_prior_counts(ds: DatasetPaths, n_std: int = 5) -> dict:
    """Validated dedicated-prior counts for one dataset at one label window.

    Split-independent, so the driver counts once per dataset and reuses the result
    across every seed and every prior-dependent ablation.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``prior_sims`` keys.
        n_std: Resolution-sigma window for the label (default ``5``, the deployed one).

    Returns:
        ``counts[bucket][class]`` from :func:`prior_pool_counts`.
    """
    return prior_pool_counts(extract_prior_pool(ds), n_std)


def deploy_prior(trainer: Trainer, counts: dict) -> Trainer:
    """Move a fitted trainer to the dedicated-prior operating point (in place).

    Thin pass-through to :func:`~pipeline.priors.apply_prior_counts`, kept so every
    prior-dependent ablation reaches the deployed decision rule by the same call
    and none silently keeps its training-count prior.

    Args:
        trainer: A fitted trainer whose bucket models receive the prior.
        counts: Counts from :func:`dedicated_prior_counts`.

    Returns:
        The same trainer, for chaining.
    """
    apply_prior_counts(trainer, counts)
    return trainer


def reference(train: dict, counts: dict | None = None) -> tuple[Trainer, dict]:
    """Fit the reference model at the fixed reference config.

    Reproduces the deployed pipeline (a default ``Trainer`` -> ``fit``), so the
    returned trainer is the reference the ablations are compared against and
    the config is reused by the model-level ablations (so only the named component
    differs between rows).

    Args:
        train: Nested per-class, per-bucket training feature dict.
        counts: Dedicated-prior counts to deploy at (:func:`dedicated_prior_counts`).
            Required for any comparison decided at the deployed rule — without it the
            trainer keeps the training split's class balance as its prior, which is a
            different operating point from the deployed pipeline. None leaves the
            training-count prior in place (only safe for prior-free scores such as
            :func:`best_f1_record` and :func:`bucket_full_scores`).

    Returns:
        ``(trainer, config)`` — the fitted trainer and its ``Trainer`` config.
    """
    trainer = Trainer()
    trainer.fit(train)
    if counts is not None:
        deploy_prior(trainer, counts)
    return trainer, trainer.config()


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
    return {**counts, **metrics(counts), **prior_free_scores(evaluator.pooled_log_ratio(data))}


def aggregate_seed_records(records: list[dict]) -> dict:
    """Collapse per-seed metric records into one mean record with ``<key>_std`` spreads.

    Every scalar entry becomes the mean of its finite per-seed values plus a
    ``<key>_std`` column holding the population std across seeds — NaN when fewer
    than two finite values exist, so a single-seed run reports no spread rather
    than a misleading zero. A single record keeps its values (and int types)
    untouched, so ``--seeds 1`` output matches a pre-aggregation run exactly.
    Non-scalar entries (e.g. the learned-weights dict) are carried from the
    first record unchanged.

    Args:
        records: One metric record per seed (all sharing the same scalar keys).

    Returns:
        The aggregated record: means under the original keys, spreads under
        ``<key>_std``, non-scalars passed through.
    """
    out: dict = {}
    for key, first in records[0].items():
        if not isinstance(first, (int, float)):
            out[key] = first
            continue
        if len(records) == 1:
            out[key] = first
            out[f"{key}_std"] = float("nan")
            continue
        finite = [float(r[key]) for r in records if not math.isnan(float(r[key]))]
        out[key] = sum(finite) / len(finite) if finite else float("nan")
        if len(finite) >= 2:
            mean = out[key]
            out[f"{key}_std"] = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))
        else:
            out[f"{key}_std"] = float("nan")
    return out


def save_model_plots(trainer: Trainer, eval_data: dict, out_dir: Path) -> None:
    """Save one fitted model's diagnostic plot set.

    Density-term heatmaps (model-level), plus the eval-split confusion matrix at
    the trainer's deployed operating point and the ROC/PR curves of its pooled
    log-ratio — written per variant so a reference and an ablation (or two
    ablations) can be compared figure by figure.

    Args:
        trainer: The variant's fitted trainer.
        eval_data: The held-out evaluation split.
        out_dir: Directory for this variant's figures (created on demand).
    """
    evaluator = Evaluator(trainer)
    save_density_terms(trainer, out_dir)
    save_confusion_matrix(pooled_counts(evaluator, eval_data), out_dir, "eval")
    save_score_curves(evaluator.pooled_log_ratio(eval_data), out_dir, "eval")


def save_score_plots(scores: torch.Tensor, labels: torch.Tensor, counts: dict, out_dir: Path) -> None:
    """Save the confusion/ROC/PR set for a variant scored outside a trainer.

    For ablations whose decision score is not the trainer's pooled log-ratio
    (e.g. the learned-weights logits): the confusion matrix comes from the
    ablation's own eval ``counts`` and the curves from its ``scores``/``labels``.

    Args:
        scores: Per-event decision scores on the eval split.
        labels: Boolean ground-truth labels aligned with ``scores``.
        counts: Confusion counts (``tp``/``fp``/``fn``/``tn``) at the variant's
            deployed operating point.
        out_dir: Directory for this variant's figures (created on demand).
    """
    save_confusion_matrix(counts, out_dir, "eval")
    save_figure(plot_roc_curve(scores, labels), Path(out_dir) / "roc_eval.png")
    save_figure(plot_pr_curve(scores, labels), Path(out_dir) / "pr_eval.png")


def best_f1_record(trainer: Trainer, train: dict, eval_data: dict) -> dict:
    """Full-model metric record at its best-F1 (train-calibrated) operating point.

    The fair reference for a baseline that *also* tunes its threshold for F1 (the learned
    weights): both then sit at their best-F1 cut on the same pooled population, so the
    comparison isolates model quality from the operating point — not the deployed
    count-prior point, which is not F1-optimized and would hand a F1-tuned baseline an
    unfair edge.

    Args:
        trainer: The fitted reference trainer.
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


def bins_from_counts(n_min: int, d: int, rho_floor: float, min_bins: int = 2) -> int:
    """Bins-per-axis sized so the scarcer class keeps ~``rho_floor`` events per cell.

    A ``d``-dimensional histogram has ``B ** d`` cells; choosing
    ``B = floor((n_min / rho_floor) ** (1 / d))`` keeps the expected count per
    cell near ``rho_floor`` for the limiting (minority) class, so the estimated
    density is not a set of memorized single-event spikes. Total cells stay
    ``O(n_min / rho_floor)`` — linear in the data, never blowing up the matrix.

    Args:
        n_min (int): Finite-feature event count of the limiting class for this term.
        d (int): Histogram dimensionality (1 or 2).
        rho_floor (float): Target events per cell.
        min_bins (int): Smallest bins-per-axis to return.

    Returns:
        int: Bins-per-axis, at least ``min_bins``.
    """
    if n_min <= 0:
        return min_bins
    return max(int((n_min / rho_floor) ** (1.0 / d)), min_bins)


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
        trainer: The fitted reference trainer.
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
        evaluator: Evaluator bound to the fitted reference trainer.
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
