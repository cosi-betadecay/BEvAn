import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.plots import plot_confusion_matrix, plot_density_matrix, plot_pr_curve, plot_roc_curve

if TYPE_CHECKING:
    from pipeline.eval import Evaluator
    from pipeline.train import Trainer


def _save_figure(fig: Figure, path: Path) -> None:
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
    _save_figure(fig, Path(out_dir) / f"confusion_matrix_{split_name}.png")


def save_score_curves(evaluator: "Evaluator", data: dict, out_dir: Path, split_name: str) -> None:
    """Save a split's ROC and PR figures (``roc_<split>.png`` / ``pr_<split>.png``).

    The local-disk counterpart of :func:`utils.wandb_logging.log_score_curves`.
    No-op when the split pools no events.

    Args:
        evaluator: A trained evaluator providing the pooled per-event log-ratio.
        data: Nested per-class, per-bucket feature dict for this split.
        out_dir: Per-dataset results directory.
        split_name: Split label (e.g. ``"train"`` or ``"eval"``).
    """
    pooled = evaluator.pooled_log_ratio(data)
    if pooled is None:
        return
    scores, labels = pooled
    out = Path(out_dir)
    _save_figure(plot_roc_curve(scores, labels), out / f"roc_{split_name}.png")
    _save_figure(plot_pr_curve(scores, labels), out / f"pr_{split_name}.png")


def save_density_terms(trainer: "Trainer", out_dir: Path) -> None:
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
            feats = term["xfeat"] if term.get("kind") == "1d" else f"{term['xfeat']}_{term['yfeat']}"
            _save_figure(plot_density_matrix(term), out / f"density_n{bucket}_{feats}.png")


def write_metrics_json(out_dir: Path, record: dict) -> Path:
    """Write ``record`` to ``<out_dir>/metrics.json`` and return the path.

    Args:
        out_dir: Per-dataset results directory.
        record: Run record — dataset provenance, the chosen ``selection`` config,
            and the per-split (``train``/``eval``) metric blocks.

    Returns:
        Path: The written ``metrics.json`` path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    with path.open("w") as fh:
        json.dump(record, fh, indent=2)
    return path
