import torch

from dataset.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.eval import Evaluator
from pipeline.train import Trainer


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


def cross_validate(
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    rho_floor: float,
    k: int = 5,
    seed: int = 0,
    **trainer_kwargs,
) -> float:
    """Mean k-fold held-out log-likelihood of the bin rule at ``rho_floor``.

    Folds are taken per class and bucket, so class balance is preserved in every
    split. For each fold a :class:`~pipeline.train.Trainer` with the given
    ``rho_floor`` is fit on the other ``k - 1`` folds and scored on the held-out
    fold with :meth:`~pipeline.eval.Evaluator.held_out_log_likelihood`. The eval
    split is never involved, so this is a leak-free generalization estimate for
    selecting the resolution. Runs in ``O(k * N)`` time and ``O(N)`` extra space.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        rho_floor: Events-per-cell constant passed to the Trainer's bin rule.
        k: Number of folds.
        seed: Seed for the reproducible fold permutation.
        **trainer_kwargs: Extra keyword arguments forwarded to :class:`Trainer`.

    Returns:
        The held-out log-likelihood averaged over the folds (NaN if none score).
    """
    generator = torch.Generator().manual_seed(seed)
    folds = {
        cls: {b: fold_indices(train[cls][b]["delta_E"].numel(), k, generator) for b in BUCKETS}
        for cls in CLASSES
    }

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
        trainer = Trainer(rho_floor=rho_floor, **trainer_kwargs).fit(tr)
        scores.append(Evaluator(trainer).held_out_log_likelihood(va))

    finite = [s for s in scores if s == s]  # drop NaN folds
    return sum(finite) / len(finite) if finite else float("nan")


# Geometric grid of events-per-cell candidates: coarse (few, large bins) to fine
# (many, small bins). Geometric spacing samples the bins-per-axis response, which
# scales as ``rho_floor ** (-1/d)``, at roughly even log-steps.
DEFAULT_RHO_FLOORS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# Laplace pseudo-counts per cell. Floored at 0 (no smoothing); negative is invalid
# (it can drive cell counts negative and corrupt the normalization). No ceiling:
# the value is applied directly, not amplified, so the grid itself is the bound.
DEFAULT_JOINT_SMOOTHINGS: tuple[float, ...] = (0.0, 0.5, 1.0)
DEFAULT_MARGINAL_SMOOTHINGS: tuple[float, ...] = (0.0, 0.5)


def select_hyperparams(
    train: dict[str, dict[int, dict[str, torch.Tensor]]],
    rho_floors: tuple[float, ...] = DEFAULT_RHO_FLOORS,
    joint_smoothings: tuple[float, ...] = DEFAULT_JOINT_SMOOTHINGS,
    marginal_smoothings: tuple[float, ...] = DEFAULT_MARGINAL_SMOOTHINGS,
    k: int = 5,
    seed: int = 0,
) -> tuple[dict[str, float], dict[tuple[float, float, float], float]]:
    """Joint grid search for the ``Trainer`` config with the best CV log-likelihood.

    Sweeps the product grid of ``(rho_floor, joint_smoothing, marginal_smoothing)``
    through :func:`cross_validate` and returns the argmax. The reward is prior-free
    density fit on held-out folds: ``rho_floor`` trades bin resolution (coarse
    underfits, fine overfits into noise), while the smoothing terms add Laplace
    pseudo-counts that keep minority-class test events out of zero-density cells.
    The eval split is never touched, so selection is leak-free.
    ``O(len(rho_floors) * len(joint_smoothings) * len(marginal_smoothings) * k * N)``
    time.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        rho_floors: Events-per-cell candidates (each must be > 0).
        joint_smoothings: Laplace pseudo-count candidates for the 2D joints (each >= 0).
        marginal_smoothings: Laplace pseudo-count candidates for the 1D marginal (each >= 0).
        k: Number of CV folds.
        seed: Seed for the reproducible fold permutation (shared across candidates).

    Returns:
        ``(best_config, scores)``. ``best_config`` is a dict with keys
        ``rho_floor``, ``joint_smoothing``, ``marginal_smoothing`` ready to splat
        into :class:`Trainer`. ``scores`` maps each ``(rho_floor, joint_smoothing,
        marginal_smoothing)`` triple to its mean held-out log-likelihood;
        NaN-scoring triples are never selected.

    Raises:
        ValueError: If any candidate grid is empty, any ``rho_floor`` is not
            positive, any smoothing is negative, or no triple scores finite.
    """
    if not (rho_floors and joint_smoothings and marginal_smoothings):
        raise ValueError("Every candidate grid must contain at least one value.")
    if any(rf <= 0 for rf in rho_floors):
        raise ValueError("rho_floor candidates must be positive.")
    if any(s < 0 for s in joint_smoothings) or any(s < 0 for s in marginal_smoothings):
        raise ValueError("smoothing candidates must be non-negative.")

    scores = {
        (rf, js, ms): cross_validate(
            train, rho_floor=rf, k=k, seed=seed, joint_smoothing=js, marginal_smoothing=ms
        )
        for rf in rho_floors
        for js in joint_smoothings
        for ms in marginal_smoothings
    }
    finite = {cfg: s for cfg, s in scores.items() if s == s}  # drop NaN candidates
    if not finite:
        raise ValueError("No hyperparameter triple produced a finite CV score.")
    rf, js, ms = max(finite, key=finite.get)
    best = {"rho_floor": rf, "joint_smoothing": js, "marginal_smoothing": ms}
    return best, scores
