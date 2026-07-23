import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.plots import (
    density_term_features,
    plot_confusion_matrix,
    plot_density_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_sky_backprojection,
)

if TYPE_CHECKING:
    import torch

    from pipeline.train import Trainer


def save_figure(fig: Figure, path: Path) -> None:
    """Save ``fig`` to ``path`` (creating parent directories) and close it.

    Args:
        fig: The matplotlib figure to write.
        path: Destination ``.png`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(counts: dict, out_dir: Path, split_name: str) -> None:
    """Save a split's confusion matrix as ``confusion_matrix_<split>.png``.

    The local-disk counterpart of
    :func:`utils.wandb_logging.log_confusion_matrix`.

    Args:
        counts: Confusion counts with ``tp``/``fp``/``fn``/``tn`` keys.
        out_dir: Per-dataset results directory.
        split_name: Split label (e.g. ``"train"`` or ``"eval"``).
    """
    fig = plot_confusion_matrix(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    save_figure(fig, Path(out_dir) / f"confusion_matrix_{split_name}.png")


def save_score_curves(pooled: tuple | None, out_dir: Path, split_name: str) -> None:
    """Save a split's ROC and PR figures (``roc_<split>.png`` / ``pr_<split>.png``).

    The local-disk counterpart of :func:`utils.wandb_logging.log_score_curves`.
    No-op when the split pools no events.

    Args:
        pooled: The split's ``(scores, labels)`` from
            :meth:`~pipeline.eval.Evaluator.pooled_log_ratio`, or None.
        out_dir: Per-dataset results directory.
        split_name: Split label (e.g. ``"train"`` or ``"eval"``).
    """
    if pooled is None:
        return
    scores, labels = pooled
    out = Path(out_dir)
    save_figure(plot_roc_curve(scores, labels), out / f"roc_{split_name}.png")
    save_figure(plot_pr_curve(scores, labels), out / f"pr_{split_name}.png")


def save_density_terms(trainer: Trainer, out_dir: Path) -> None:
    """Save every fitted density term as ``density_n<bucket>_<feats>.png``.

    The local-disk counterpart of :func:`utils.wandb_logging.log_density_terms`.
    Densities are model-level (independent of split), so this is called once per run.

    Args:
        trainer: A fitted trainer with per-bucket ``models``.
        out_dir: Per-dataset results directory.
    """
    out = Path(out_dir)
    for bucket, model in trainer.models.items():
        for term in model.get("terms", []):
            feats = density_term_features(term)
            save_figure(plot_density_matrix(term), out / f"density_n{bucket}_{feats}.png")


def save_sky_backprojection(
    image: torch.Tensor,
    n_events: int,
    out_dir: Path,
    phi_range: tuple[float, float],
    theta_range: tuple[float, float],
) -> None:
    """Save the sky-image figure as ``sky_backprojection.png``.

    The local-disk counterpart of
    :func:`utils.wandb_logging.log_sky_backprojection`. Model-level
    (independent of split), so this is called once per run.

    Args:
        image: The accumulated ``(n_theta, n_phi)`` sky image (see
            :func:`utils.plots.plot_sky_backprojection`).
        n_events: Number of events stacked into the image.
        out_dir: Per-dataset results directory.
        phi_range: ``(min, max)`` azimuthal extent of the image, in radians.
        theta_range: ``(min, max)`` polar extent of the image, in radians.
    """
    fig = plot_sky_backprojection(image, n_events, phi_range, theta_range)
    save_figure(fig, Path(out_dir) / "sky_backprojection.png")


def write_metrics_json(out_dir: Path, record: dict) -> Path:
    """Write ``record`` to ``<out_dir>/metrics.json`` and return the path.

    Args:
        out_dir: Per-dataset results directory.
        record: Run record — dataset provenance, the chosen ``selection`` config,
            and the per-split (``train``/``eval``) metric blocks.

    Returns:
        The written ``metrics.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    with path.open("w") as fh:
        json.dump(record, fh, indent=2)
    return path
