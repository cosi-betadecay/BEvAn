from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import wandb
from matplotlib.figure import Figure

from utils.plots import (
    density_term_features,
    plot_confusion_matrix,
    plot_density_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

if TYPE_CHECKING:
    from pipeline.eval import Evaluator
    from pipeline.train import Trainer


def _log_figure(key: str, fig: Figure) -> None:
    """Attach a figure to the active W&B run under ``key``, then close it.

    Mirrors the image-logging half of
    :func:`modeling.calculate_probablities.log_confusion`. Assumes a run is active
    (the public helpers guard on ``wandb.run``) and always closes the figure so
    repeated logging never leaks matplotlib state.

    Args:
        key: W&B metric key to log the image under.
        fig: The matplotlib figure to log.
    """
    wandb.log({key: wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix(counts: dict, split_name: str = "eval") -> None:
    """Log a confusion-matrix image to W&B from raw confusion counts.

    The image-logging half of
    :func:`modeling.calculate_probablities.log_confusion`, centralized here
    alongside the other figure loggers; that function delegates to this one. No-op
    when no run is active.

    Args:
        counts: Confusion counts with ``tp``/``fp``/``fn``/``tn`` keys.
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"eval/n2"``).
    """
    if wandb.run is None:
        return
    fig = plot_confusion_matrix(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    _log_figure(f"{split_name}/confusion_matrix", fig)


def log_roc_curve(scores: object, labels: object, split_name: str = "eval") -> None:
    """Log a ROC-curve image to W&B, the way the confusion matrix is logged.

    No-op when no run is active, so callers need no guard of their own.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"train"``).
    """
    if wandb.run is None:
        return
    _log_figure(f"{split_name}/roc_curve", plot_roc_curve(scores, labels))


def log_pr_curve(scores: object, labels: object, split_name: str = "eval") -> None:
    """Log a precision-recall-curve image to W&B, the way the confusion matrix is logged.

    No-op when no run is active, so callers need no guard of their own.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"train"``).
    """
    if wandb.run is None:
        return
    _log_figure(f"{split_name}/pr_curve", plot_pr_curve(scores, labels))


def log_density_matrix(term: dict, split_name: str = "model", bucket: int | None = None) -> None:
    """Log a fitted density-term figure to W&B, the way the confusion matrix is logged.

    No-op when no run is active. The key is namespaced by split, optional
    hit-multiplicity bucket (matching the ``"{split}/n{bucket}"`` scheme the
    confusion logging uses), and the term's feature(s), so the same feature pair
    in different buckets does not collide.

    Args:
        term: A 1D or 2D density term dict from a fitted model.
        split_name: Metric namespace prefix; densities are model-level, so
            ``"model"`` by default (they do not depend on the eval split).
        bucket: Optional hit-multiplicity bucket, folded into the key when given.
    """
    if wandb.run is None:
        return
    feats = density_term_features(term)
    scope = f"{split_name}/density" + (f"/n{bucket}" if bucket is not None else "")
    _log_figure(f"{scope}/{feats}", plot_density_matrix(term))


def log_score_curves(evaluator: "Evaluator", data: dict, split_name: str = "eval") -> None:
    """Log the ROC and precision-recall curves for one split's pooled scores.

    Pulls the pooled per-event log-ratio and truths from ``evaluator`` (the same
    score the prior-free metrics use) and logs both curves. No-op when no run is
    active or the split pools no events.

    Args:
        evaluator: A trained :class:`~pipeline.eval.Evaluator`.
        data: Nested per-class, per-bucket feature dict for this split.
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"train"``).
    """
    if wandb.run is None:
        return
    pooled = evaluator.pooled_log_ratio(data)
    if pooled is None:
        return
    scores, labels = pooled
    log_roc_curve(scores, labels, split_name)
    log_pr_curve(scores, labels, split_name)


def log_density_terms(trainer: "Trainer", split_name: str = "model") -> None:
    """Log every fitted density term across the trainer's per-bucket models.

    Densities are model-level (independent of the eval split), so this is meant
    to be called once per trained model. No-op when no run is active.

    Args:
        trainer: A fitted :class:`~pipeline.train.Trainer`.
        split_name: Metric namespace prefix for the density keys.
    """
    if wandb.run is None:
        return
    for bucket, model in trainer.models.items():
        for term in model.get("terms", []):
            log_density_matrix(term, split_name, bucket=bucket)
