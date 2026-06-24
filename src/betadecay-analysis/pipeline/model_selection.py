import math

import torch
from tqdm import tqdm

from dataset.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.eval import Evaluator, best_f1_threshold, f1_at_threshold
from pipeline.train import Trainer

# Absolute floor (in F1) the calibrated threshold offset must beat, on top of the
# 1-SE guard, before it is adopted. Keeps the champion's ~0.001 eval-overfit gap
# from ever moving the operating point (so champion runs stay byte-identical).
CALIB_FLOOR: float = 0.003
# Absolute floor (in AUC) a searched config's confirmation-split gain must beat,
# on top of the extreme-value-scaled SE bar, before it displaces the champion seed.
# AUC is saturated (~0.99) on these geometries, so sub-floor "gains" are fold noise:
# a 1000-iter run without this floor adopted a noise-winner that cost Max 0.025 F1.
SEARCH_FLOOR: float = 0.002

# The hardcoded per-bucket bin budget that is the main-branch champion (best F1
# on Max). The search is seeded to reproduce these bins on each dataset's own
# event counts, so the champion is the starting point and stays reachable.
CHAMPION_BINS: dict[int, int] = {1: 25, 2: 8, 3: 35}
# Histogram dimensionality of each bucket's primary density term (bucket 1 is the
# 1D delta_E marginal; buckets 2 and 3 lead with a 2D joint).
BUCKET_DIM: dict[int, int] = {1: 1, 2: 2, 3: 2}


def fold_indices(n: int, k: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Partition ``range(n)`` into ``k`` near-equal random folds.

    Args:
        n: Number of items to split.
        k: Number of folds.
        generator: Seeded generator for the reproducible permutation.

    Returns:
        ``k`` disjoint index tensors whose union is a permutation of ``range(n)``.
    """
    perm = torch.randperm(n, generator=generator)
    return [perm[i::k] for i in range(k)]


def _finite_count(col_dict: dict[str, torch.Tensor], *feats: str) -> int:
    """Number of rows that are finite across every named feature column."""
    mask = torch.isfinite(col_dict[feats[0]])
    for f in feats[1:]:
        mask = mask & torch.isfinite(col_dict[f])
    return int(mask.sum())


def _rho_for_bins(n_min: int, d: int, bins: int) -> float:
    """Invert :func:`~modeling.matrix_calculations.bins_from_counts`.

    That rule sets ``bins = floor((n_min / rho) ** (1 / d))``, so the ``rho`` that
    reproduces a target ``bins`` for a class with ``n_min`` limiting events is
    ``n_min / bins ** d``. The ``1 - 1e-9`` nudge keeps the value a hair above the
    integer boundary so the ``floor`` lands on ``bins`` rather than ``bins - 1``
    from float rounding (far too small to ever reach ``bins + 1``). Falls back to
    ``1.0`` when undefined (empty class).
    """
    if n_min <= 0 or bins <= 0:
        return 1.0
    return n_min / (bins**d) * (1 - 1e-9)


def champion_seed(
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    n_bins: dict[int, int] | None = None,
    joint_smoothing: float = 0.5,
    marginal_smoothing: float = 0.0,
) -> dict:
    """Per-bucket ``rho_floor`` config that reproduces ``n_bins`` on ``train``.

    Computes, for each bucket, the events-per-cell value that makes the dynamic
    bin rule emit the champion's bin count given this dataset's own limiting
    (scarcer-class) finite counts. The result is a ``Trainer``-ready config dict:
    on the seeding dataset it reproduces the hardcoded ``{1:25, 2:8, 3:35}`` exactly,
    but because it is expressed as ``rho_floor`` the resolution still scales with
    event count on any other geometry — dynamic in form, champion in value.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        n_bins: Per-bucket target bin counts; defaults to :data:`CHAMPION_BINS`.
        joint_smoothing: Seed Laplace pseudo-counts for the 2D joints.
        marginal_smoothing: Seed Laplace pseudo-counts for the 1D marginal.

    Returns:
        ``{"rho_floor": {bucket: rho}, "joint_smoothing": ..., "marginal_smoothing": ...}``.
    """
    n_bins = n_bins or CHAMPION_BINS
    bd, bg = train["bdecay"], train["bg"]
    # Bucket 1 leads with the 1D delta_E marginal; 2 and 3 with the 2D (delta_E, arm) joint.
    feats = {1: ("delta_E",), 2: ("delta_E", "arm"), 3: ("delta_E", "arm")}
    rho = {
        b: _rho_for_bins(
            min(_finite_count(bd[b], *feats[b]), _finite_count(bg[b], *feats[b])),
            BUCKET_DIM[b],
            n_bins[b],
        )
        for b in BUCKETS
    }
    return {
        "rho_floor": rho,
        "joint_smoothing": joint_smoothing,
        "marginal_smoothing": marginal_smoothing,
    }


def flatten_config(config: dict) -> dict[str, float]:
    """Flatten a config dict to scalar columns for logging / CSV output."""
    rho = config["rho_floor"]
    if isinstance(rho, dict):
        out = {f"rho_floor_{b}": float(rho[b]) for b in sorted(rho)}
    else:
        out = {"rho_floor": float(rho)}
    out["joint_smoothing"] = float(config["joint_smoothing"])
    out["marginal_smoothing"] = float(config["marginal_smoothing"])
    return out


def _mean_se(scores: list[float]) -> tuple[float, float]:
    """Sample mean and standard error of the mean (inf SE if under two points)."""
    n = len(scores)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(scores) / n
    if n < 2:
        return mean, float("inf")
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    return mean, math.sqrt(var / n)


def cv_fold_scores(
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    config: dict,
    k: int = 5,
    seed: int = 0,
    reward: str = "auc",
) -> list[float]:
    """Held-out reward of ``config`` on each of ``k`` leak-free folds.

    Folds are taken per class and bucket, so class balance is preserved in every
    split. For each fold a :class:`~pipeline.train.Trainer` built from ``config``
    is fit on the other ``k - 1`` folds and scored on the held-out fold. The eval
    split is never involved.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        config: ``Trainer`` kwargs (``rho_floor`` may be a per-bucket dict).
        k: Number of folds.
        seed: Seed for the reproducible fold permutation.
        reward: ``"auc"`` for separating power (default) or ``"loglik"`` for the
            held-out log-likelihood.

    Returns:
        The finite per-fold scores (NaN folds dropped).
    """
    generator = torch.Generator().manual_seed(seed)
    folds = {
        cls: {b: fold_indices(train[cls][b]["delta_E"].numel(), k, generator) for b in BUCKETS}
        for cls in CLASSES
    }
    metric = "held_out_auc" if reward == "auc" else "held_out_log_likelihood"

    scores = []
    for f in range(k):
        tr = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
        va = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
        for cls in CLASSES:
            for b in BUCKETS:
                n = train[cls][b]["delta_E"].numel()
                val_idx = folds[cls][b][f]
                keep = torch.ones(n, dtype=torch.bool)
                keep[val_idx] = False
                for feat in FEATURES:
                    col = train[cls][b][feat]
                    tr[cls][b][feat] = col[keep]
                    va[cls][b][feat] = col[val_idx]
        trainer = Trainer(**config).fit(tr)
        scores.append(getattr(Evaluator(trainer), metric)(va))

    return [s for s in scores if s == s]  # drop NaN folds


def _sample_config(
    seed_config: dict,
    generator: torch.Generator,
    rho_factor: float,
    joint_range: tuple[float, float],
    marginal_range: tuple[float, float],
) -> dict:
    """Draw one candidate near the seed: log-uniform per-bucket rho, uniform smoothing."""

    def log_uniform(center: float) -> float:
        u = float(torch.rand(1, generator=generator)) * 2 - 1  # [-1, 1]
        return max(center * math.exp(u * math.log(rho_factor)), 1e-6)

    def uniform(lo: float, hi: float) -> float:
        return lo + float(torch.rand(1, generator=generator)) * (hi - lo)

    return {
        "rho_floor": {b: log_uniform(seed_config["rho_floor"][b]) for b in BUCKETS},
        "joint_smoothing": uniform(*joint_range),
        "marginal_smoothing": uniform(*marginal_range),
    }


def search_hyperparams(
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    n_iter: int = 1000,
    k: int = 5,
    seed: int = 0,
    reward: str = "auc",
    floor: float = SEARCH_FLOOR,
    n_bins: dict[int, int] | None = None,
    joint_smoothing: float = 0.5,
    marginal_smoothing: float = 0.0,
    rho_factor: float = 4.0,
    joint_range: tuple[float, float] = (0.0, 2.0),
    marginal_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict, dict]:
    """Champion-seeded random search for the bin + smoothing config, overfit-guarded.

    Starts from :func:`champion_seed` (the hardcoded ``{1:25, 2:8, 3:35}`` expressed
    as per-bucket ``rho_floor``) and draws ``n_iter`` candidates around it, each
    scored by leak-free k-fold held-out **AUC** — separating power, the quantity F1
    tracks, not the misaligned log-likelihood. The best-scoring candidate on the
    search split is then re-scored on an *independent* fold split (``seed + 1``)
    against the seed; it is only adopted if its margin beats both an extreme-value
    bar ``z * pooled_SE`` with ``z = sqrt(2 ln n_iter)`` (the multiple-comparisons
    correction for the winner's curse over ``n_iter`` draws) and an absolute
    ``floor``. Otherwise the champion seed is returned. A plain one-SE gate proved
    too weak: at ``n_iter=1000`` it adopted a 0.001-AUC noise-winner that cost the
    Max champion 0.025 F1; the scaled bar plus floor keeps the champion frozen
    unless a gain is large and real. The eval split is never touched.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        n_iter: Number of random candidates to try around the seed.
        k: Number of CV folds.
        seed: Base seed; the search split uses it, the confirmation split ``seed + 1``.
        reward: ``"auc"`` (default) or ``"loglik"``.
        floor: Absolute AUC gain the winner must beat (on top of the scaled-SE bar).
        n_bins: Champion bin budget to seed from; defaults to :data:`CHAMPION_BINS`.
        joint_smoothing: Seed joint smoothing.
        marginal_smoothing: Seed marginal smoothing.
        rho_factor: Per-bucket rho is sampled log-uniformly in ``[rho/factor, rho*factor]``.
        joint_range: Uniform sampling range for joint smoothing.
        marginal_range: Uniform sampling range for marginal smoothing.

    Returns:
        ``(chosen_config, info)``. ``chosen_config`` is ``Trainer``-ready. ``info``
        records the seed config, whether the search winner was accepted, and the
        confirmation-split means / margin / pooled SE behind that decision.

    Raises:
        ValueError: If the champion seed itself scores no finite fold (degenerate split).
    """
    seed_config = champion_seed(train, n_bins, joint_smoothing, marginal_smoothing)
    generator = torch.Generator().manual_seed(seed)

    # --- Search split: pick the best-mean candidate (the seed is the baseline). ---
    best_config = seed_config
    best_mean, _ = _mean_se(cv_fold_scores(train, seed_config, k, seed, reward))
    if best_mean != best_mean:
        raise ValueError("Champion seed produced no finite CV fold; cannot search.")
    pbar = tqdm(range(n_iter), desc="hyperparam search", unit="cfg")
    pbar.set_postfix(best_auc=f"{best_mean:.4f}")
    for _ in pbar:
        candidate = _sample_config(seed_config, generator, rho_factor, joint_range, marginal_range)
        mean, _ = _mean_se(cv_fold_scores(train, candidate, k, seed, reward))
        if mean == mean and mean > best_mean:
            best_config, best_mean = candidate, mean
            pbar.set_postfix(best_auc=f"{best_mean:.4f}")

    # --- Confirmation split: adopt the winner only on convincing, real evidence. ---
    # With n_iter draws the search-split argmax is winner's-curse inflated, and AUC
    # here is saturated (~0.99) so sub-floor "gains" are fold noise. Require the
    # confirmation-split margin to beat BOTH an extreme-value-scaled SE bar
    # (z ~ sqrt(2 ln n_iter), the expected max of n_iter noise draws) and an
    # absolute floor, so the champion is frozen unless a gain is large and real.
    c_seed_mean, c_seed_se = _mean_se(cv_fold_scores(train, seed_config, k, seed + 1, reward))
    c_best_mean, c_best_se = _mean_se(cv_fold_scores(train, best_config, k, seed + 1, reward))
    margin = c_best_mean - c_seed_mean
    pooled_se = math.sqrt(c_best_se**2 + c_seed_se**2)
    z = max(1.0, math.sqrt(2.0 * math.log(max(n_iter, 2))))
    bar = max(z * pooled_se, floor)
    accepted = (best_config is not seed_config) and (margin == margin) and (margin > bar)
    chosen = best_config if accepted else seed_config

    info = {
        "accepted": accepted,
        "reward": reward,
        "n_iter": n_iter,
        "search_best_mean": best_mean,
        "confirm_seed_mean": c_seed_mean,
        "confirm_best_mean": c_best_mean,
        "confirm_margin": margin,
        "confirm_pooled_se": pooled_se,
        "accept_bar": bar,
        "seed_config": seed_config,
        "chosen_config": chosen,
    }
    return chosen, info


def _pooled_margins(
    trainer: Trainer, data: dict[str, dict[int, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-event decision margin ``LLR - log(n_bg/n_beta)`` pooled across buckets.

    The default Bayesian rule predicts β⁺ exactly when this margin is ``>= 0``, so
    a single global cut ``c`` on the pooled margin is the one knob that shifts every
    bucket's operating point off the count-prior default while keeping each bucket's
    own prevalence baked into its baseline. Buckets with an empty class (no finite
    margin baseline) are skipped.

    Args:
        trainer: A fitted trainer providing per-bucket models and log-ratios.
        data: Nested per-class, per-bucket feature dict for one split.

    Returns:
        ``(margins, labels)`` over all calibratable buckets (empty tensors if none).
    """
    evaluator = Evaluator(trainer)
    margins, labels = [], []
    for b in BUCKETS:
        model = trainer.models[b]
        if model["n_beta"] <= 0 or model["n_bg"] <= 0:
            continue  # degenerate prior-only bucket: no count-prior baseline to shift
        out = evaluator.bucket_log_ratio(data, b, model)
        if out is None:
            continue
        llr, gt = out
        tau = math.log(model["n_bg"] / model["n_beta"])
        margins.append(llr - tau)
        labels.append(gt)
    if not margins:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    return torch.cat(margins), torch.cat(labels)


def calibrate_global_offset(
    trainer: Trainer,
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    config: dict,
    k: int = 5,
    seed: int = 0,
    floor: float = CALIB_FLOOR,
) -> tuple[float, dict]:
    """Find the global decision-threshold offset that maximizes pooled F1, guarded.

    The deployed decision predicts β⁺ when the pooled margin ``>= c``; ``c = 0`` is
    today's count-prior operating point. The F1-optimal ``c`` is read off the
    *full-train* model (scale-correct for what ships) via :func:`best_f1_threshold`.
    Whether to actually move there is then decided leak-free: across ``k`` folds the
    held-out pooled F1 at the candidate ``c`` is compared to ``c = 0``, and the
    offset is adopted only if the mean gain beats both one standard error and an
    absolute ``floor``. The floor keeps the champion's ~0.001 eval-overfit gap from
    moving anything, so champion configs stay byte-identical; a real miscalibration
    (e.g. a heavily-smoothed config whose operating point drifted) clears it.

    Args:
        trainer: The deployed model, already fit on the full training split.
        train: Nested per-class, per-bucket training feature dict.
        config: The ``Trainer`` config used (re-fit per fold for the guard).
        k: Number of CV folds for the guard.
        seed: Seed for the reproducible fold permutation.
        floor: Absolute F1 gain the offset must beat (on top of the 1-SE guard).

    Returns:
        ``(offset, info)``. ``offset`` is ``0.0`` when the default is kept; ``info``
        records the candidate offset, acceptance, and the CV gain / SE behind it.
    """
    margins, labels = _pooled_margins(trainer, train)
    if margins.numel() == 0 or int(labels.sum()) == 0:
        return 0.0, {"accepted": False, "offset": 0.0, "reason": "no calibratable events"}

    _, candidate = best_f1_threshold(margins, labels)  # margin cut maximizing in-sample F1
    if candidate != candidate or candidate == 0.0:
        return 0.0, {"accepted": False, "offset": 0.0, "candidate": candidate}

    # Leak-free guard: does moving to `candidate` beat the default on held-out folds?
    generator = torch.Generator().manual_seed(seed)
    folds = {
        cls: {b: fold_indices(train[cls][b]["delta_E"].numel(), k, generator) for b in BUCKETS}
        for cls in CLASSES
    }
    gains = []
    for f in range(k):
        tr = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
        va = {cls: {b: {} for b in BUCKETS} for cls in CLASSES}
        for cls in CLASSES:
            for b in BUCKETS:
                n = train[cls][b]["delta_E"].numel()
                val_idx = folds[cls][b][f]
                keep = torch.ones(n, dtype=torch.bool)
                keep[val_idx] = False
                for feat in FEATURES:
                    col = train[cls][b][feat]
                    tr[cls][b][feat] = col[keep]
                    va[cls][b][feat] = col[val_idx]
        fold_trainer = Trainer(**config).fit(tr)
        mv, lv = _pooled_margins(fold_trainer, va)
        if mv.numel() == 0:
            continue
        f1_default = f1_at_threshold(mv, lv, 0.0)
        f1_candidate = f1_at_threshold(mv, lv, candidate)
        if f1_default == f1_default and f1_candidate == f1_candidate:
            gains.append(f1_candidate - f1_default)

    mean_gain, se = _mean_se(gains)
    accepted = (mean_gain == mean_gain) and (mean_gain > max(se, floor))
    offset = candidate if accepted else 0.0
    info = {
        "accepted": accepted,
        "offset": offset,
        "candidate": candidate,
        "cv_mean_gain": mean_gain,
        "cv_gain_se": se,
        "floor": floor,
    }
    return offset, info


def apply_offset(trainer: Trainer, offset: float) -> None:
    """Write a global margin offset into a fitted trainer's per-bucket thresholds.

    Sets each calibratable bucket's ``log_threshold`` to ``log(n_bg/n_beta) + offset``
    so the deployed decision predicts β⁺ when ``LLR >= log(n_bg/n_beta) + offset``
    (the pooled-margin ``>= offset`` rule). An ``offset`` of 0 is a no-op: the models
    keep their default (``log_threshold`` absent -> count-prior posterior rule), so
    the run stays byte-identical to the uncalibrated model.

    Args:
        trainer: A fitted trainer to annotate in place.
        offset: The global margin offset from :func:`calibrate_global_offset`.
    """
    if offset == 0.0:
        return
    for b in BUCKETS:
        model = trainer.models[b]
        if model["n_beta"] > 0 and model["n_bg"] > 0:
            model["log_threshold"] = math.log(model["n_bg"] / model["n_beta"]) + offset
