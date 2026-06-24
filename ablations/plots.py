from pathlib import Path

import matplotlib

# Headless Agg backend before pyplot, same as utils.plots: these figures are only
# ever saved, and the study runs without a display on the production VM.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Default output directory: ablations/figures (created on first save).
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Palette reused from utils.plots so the ablation figures share the project look.
COLOR_CHAMPION = "#3b6e8c"  # champion bars (mako teal)
COLOR_ABLATION = "#e8843c"  # ablation bars (amber)
COLOR_REFERENCE = "#9aa0a6"  # dashed reference lines (neutral grey)

# The four metrics every comparison plot shows, as (record key, axis label).
COMPARISON_METRICS = (("f1_score", "F1"), ("auc", "AUC"), ("precision", "Precision"), ("recall", "Recall"))

# Factor display order and labels for the contribution panel.
FACTOR_ORDER = ("delta_E", "arm", "anni")
FACTOR_LABELS = {"delta_E": "ΔE", "arm": "ARM", "anni": "anni"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and population std of the finite entries (``(nan, 0)`` if none)."""
    arr = np.array([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    return float(arr.mean()), float(arr.std())


def _save(fig: Figure, path: Path) -> Path:
    """Save ``fig`` to ``path`` (creating parents), close it, and return the path.

    Args:
        fig: The matplotlib figure to write.
        path: Destination ``.png`` path.

    Returns:
        Path: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float]]],
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot an ablation against the champion, averaged across datasets.

    One grouped bar pair (champion vs ablation) per metric, with error bars giving
    the std across datasets, so a regressing ablation reads as the amber bar
    dropping below the teal one.

    Args:
        ablation_name: Label for the ablation series and the output filename.
        per_dataset: ``{dataset: {"champion": record, "ablation": record}}`` where
            each record carries the :data:`COMPARISON_METRICS` keys.
        save_dir: Directory to write the figure into.

    Returns:
        Path: The written ``<ablation_name>.png``.
    """
    records = list(per_dataset.values())
    x = np.arange(len(COMPARISON_METRICS))
    width = 0.38

    champion = [
        _mean_std([r["champion"].get(key, float("nan")) for r in records]) for key, _ in COMPARISON_METRICS
    ]
    ablation = [
        _mean_std([r["ablation"].get(key, float("nan")) for r in records]) for key, _ in COMPARISON_METRICS
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        [m for m, _ in champion],
        width,
        yerr=[s for _, s in champion],
        capsize=4,
        label="champion",
        color=COLOR_CHAMPION,
    )
    ax.bar(
        x + width / 2,
        [m for m, _ in ablation],
        width,
        yerr=[s for _, s in ablation],
        capsize=4,
        label=ablation_name,
        color=COLOR_ABLATION,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in COMPARISON_METRICS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean ± std across datasets)")
    ax.set_title(f"{ablation_name} vs champion (avg across datasets)")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return _save(fig, Path(save_dir) / f"{ablation_name}.png")


def plot_factor_contributions(
    contributions_by_dataset: dict[str, dict[str, dict[str, float]]],
    full_model_by_dataset: dict[str, dict[str, float]] | None = None,
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot each factor's standalone F1 and AUC, averaged across datasets.

    Two panels (F1 left, AUC right); within each, one bar per factor (mean across
    datasets, std as error bars). With ``full_model_by_dataset`` a dashed line marks
    the mean full-model metric, so a factor reaching the line carries most of the
    separating power.

    Args:
        contributions_by_dataset: ``{dataset: {factor: {"f1": ..., "auc": ...}}}``.
        full_model_by_dataset: Optional ``{dataset: {"f1": ..., "auc": ...}}``
            full-model references, averaged for the dashed line.
        save_dir: Directory to write the figure into.

    Returns:
        Path: The written ``factor_contributions.png``.
    """
    factors = [f for f in FACTOR_ORDER if any(f in contrib for contrib in contributions_by_dataset.values())]
    x = np.arange(len(factors))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (key, label) in zip(axes, (("f1", "F1"), ("auc", "AUC")), strict=True):
        stats = [
            _mean_std([c[f][key] for c in contributions_by_dataset.values() if f in c and key in c[f]])
            for f in factors
        ]
        ax.bar(
            x, [m for m, _ in stats], width=0.6, yerr=[s for _, s in stats], capsize=4, color=COLOR_CHAMPION
        )
        if full_model_by_dataset:
            full_mean, _ = _mean_std([fm.get(key, float("nan")) for fm in full_model_by_dataset.values()])
            ax.axhline(full_mean, ls="--", lw=1, color=COLOR_REFERENCE, label="full model (avg)")
            ax.legend(loc="lower right", fontsize="small")
        ax.set_xticks(x)
        ax.set_xticklabels([FACTOR_LABELS[f] for f in factors])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle("Factor contributions (avg across datasets)")
    fig.tight_layout()
    return _save(fig, Path(save_dir) / "factor_contributions.png")
