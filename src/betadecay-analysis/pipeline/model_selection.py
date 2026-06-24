import math

import torch

from dataset.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.eval import Evaluator
from pipeline.train import Trainer

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
    against the seed; it is only adopted if it still beats the champion by more
    than one pooled standard error. Otherwise the champion seed is returned. This
    one-SE-over-seed gate makes the search safe at large ``n_iter``: fold noise and
    the winner's-curse cannot clear the bar, so the model never drifts off the
    champion without reproducible evidence. The eval split is never touched.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        n_iter: Number of random candidates to try around the seed.
        k: Number of CV folds.
        seed: Base seed; the search split uses it, the confirmation split ``seed + 1``.
        reward: ``"auc"`` (default) or ``"loglik"``.
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
    for _ in range(n_iter):
        candidate = _sample_config(seed_config, generator, rho_factor, joint_range, marginal_range)
        mean, _ = _mean_se(cv_fold_scores(train, candidate, k, seed, reward))
        if mean == mean and mean > best_mean:
            best_config, best_mean = candidate, mean

    # --- Confirmation split: only leave the champion on >1 pooled-SE evidence. ---
    c_seed_mean, c_seed_se = _mean_se(cv_fold_scores(train, seed_config, k, seed + 1, reward))
    c_best_mean, c_best_se = _mean_se(cv_fold_scores(train, best_config, k, seed + 1, reward))
    margin = c_best_mean - c_seed_mean
    pooled_se = math.sqrt(c_best_se**2 + c_seed_se**2)
    accepted = (best_config is not seed_config) and (margin == margin) and (margin > pooled_se)
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
        "seed_config": seed_config,
        "chosen_config": chosen,
    }
    return chosen, info
