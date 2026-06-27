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
COLOR_MODEL = "#3b6e8c"  # our-model bars (mako teal)
COLOR_ABLATION = "#e8843c"  # ablation bars (amber)
COLOR_REFERENCE = "#9aa0a6"  # dashed reference lines (neutral grey)

# The four metrics every comparison plot shows, as (record key, axis label).
COMPARISON_METRICS = (("f1_score", "F1"), ("auc", "AUC"), ("precision", "Precision"), ("recall", "Recall"))

# Factor display order and labels for the contribution panel.
FACTOR_ORDER = ("delta_E", "arm", "anni")
FACTOR_LABELS = {"delta_E": "ΔE", "arm": "ARM", "anni": "Annihilation angle"}

# Human-readable display names for the ablations (the file stems stay machine-named).
ABLATION_TITLES = {
    "no_calibration": "No calibration",
    "learned_weights": "Learned weights",
    "no_ckd_order": "No CKD ordering",
    "factor_contributions": "Factor contributions",
    "gt_tolerance": "Ground-truth tolerance",
}

# The deployed label window (resolution sigmas); marked on the sweep so the champion
# anchor is visible against the rest of the curve.
DEPLOYED_N_STD = 3

# Font sizes tuned for figures embedded in the paper (well above the matplotlib
# defaults so titles, axes, and legends stay legible at print size).
SUPTITLE_SIZE = 18
TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13


def _pretty(name: str) -> str:
    """Human-readable display name for an ablation (underscores spaced, capitalized)."""
    return ABLATION_TITLES.get(name, name.replace("_", " ").capitalize())


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and population std of the finite entries (``(nan, 0)`` if none)."""
    arr = np.array([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    return float(arr.mean()), float(arr.std())


def _save(fig: Figure, path: Path) -> Path:
    """Save ``fig`` to ``path`` at print resolution (creating parents), close it, return it.

    Args:
        fig: The matplotlib figure to write.
        path: Destination ``.png`` path.

    Returns:
        Path: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float]]],
    save_dir: Path = FIGURES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Plot an ablation against our model, averaged across datasets.

    One grouped bar pair (our model vs ablation) per metric, with error bars giving
    the std across datasets, so a regressing ablation reads as the amber bar
    dropping below the teal one.

    Args:
        ablation_name: Ablation key; humanized for the title/legend, kept as the
            output filename stem.
        per_dataset: ``{dataset: {"our_model": record, "ablation": record}}`` where
            each record carries the :data:`COMPARISON_METRICS` keys.
        save_dir: Directory to write the figure into.
        exclude: Dataset names to drop from the average (e.g. the out-of-domain
            balloon); they stay in the per-dataset CSV but are not pooled here.

    Returns:
        Path: The written ``<ablation_name>.png``.
    """
    records = [v for k, v in per_dataset.items() if k not in exclude]
    x = np.arange(len(COMPARISON_METRICS))
    width = 0.38

    our_model = [
        _mean_std([r["our_model"].get(key, float("nan")) for r in records]) for key, _ in COMPARISON_METRICS
    ]
    ablation = [
        _mean_std([r["ablation"].get(key, float("nan")) for r in records]) for key, _ in COMPARISON_METRICS
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        [m for m, _ in our_model],
        width,
        yerr=[s for _, s in our_model],
        capsize=4,
        label="Our model",
        color=COLOR_MODEL,
    )
    ax.bar(
        x + width / 2,
        [m for m, _ in ablation],
        width,
        yerr=[s for _, s in ablation],
        capsize=4,
        label=_pretty(ablation_name),
        color=COLOR_ABLATION,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in COMPARISON_METRICS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Metric", fontsize=LABEL_SIZE)
    ax.set_ylabel("Score (mean ± std across datasets)", fontsize=LABEL_SIZE)
    ax.set_title(f"{_pretty(ablation_name)} vs. our model", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return _save(fig, Path(save_dir) / f"{ablation_name}.png")


def plot_gt_tolerance(
    results_by_dataset: dict[str, dict[int, dict[str, float]]],
    save_dir: Path = FIGURES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Plot each metric versus the label window ``n_std``, averaged across datasets.

    A 2x2 grid (F1, AUC, precision, recall); within each panel the line is the mean
    across the non-``exclude`` datasets with a ±std band, and a dashed vertical marks
    the deployed window (:data:`DEPLOYED_N_STD`). Roughly flat curves in the same band
    as the marked point are the robustness claim: the study does not hinge on the cut.

    Args:
        results_by_dataset: ``{dataset: {n_std: metric_record}}``.
        save_dir: Directory to write the figure into.
        exclude: Dataset names dropped from the average (e.g. the out-of-domain
            balloon); they stay in the per-dataset CSV but are not pooled here.

    Returns:
        Path: The written ``gt_tolerance.png``.
    """
    included = {k: v for k, v in results_by_dataset.items() if k not in exclude}
    n_std_values = sorted({n for per_n in included.values() for n in per_n})
    x = np.array(n_std_values, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, label) in zip(axes.flat, COMPARISON_METRICS, strict=True):
        means, stds = [], []
        for n_std in n_std_values:
            mean, std = _mean_std(
                [per_n[n_std].get(key, float("nan")) for per_n in included.values() if n_std in per_n]
            )
            means.append(mean)
            stds.append(std)
        means_arr, stds_arr = np.array(means), np.array(stds)
        ax.plot(x, means_arr, marker="o", color=COLOR_MODEL, label="Mean across datasets")
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, color=COLOR_MODEL, alpha=0.2)
        ax.axvline(
            DEPLOYED_N_STD, ls="--", lw=1.5, color=COLOR_REFERENCE, label=f"Deployed ({DEPLOYED_N_STD}σ)"
        )
        ax.set_ylim(0, 1)
        ax.set_xticks(n_std_values)
        ax.set_xlabel("Label window n_std (σ)", fontsize=LABEL_SIZE)
        ax.set_ylabel(label, fontsize=LABEL_SIZE)
        ax.set_title(label, fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.legend(loc="lower right", fontsize=LEGEND_SIZE)

    fig.suptitle("Robustness to the 511 keV label window (averaged across datasets)", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return _save(fig, Path(save_dir) / "gt_tolerance.png")


def plot_factor_contributions(
    contributions_by_dataset: dict[str, dict[str, dict[str, float]]],
    full_model_by_dataset: dict[str, dict[str, float]] | None = None,
    save_dir: Path = FIGURES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Plot each factor's standalone F1 and AUC, averaged across datasets.

    Two panels (F1 left, AUC right); within each, one bar per factor (mean across
    datasets, std as error bars). With ``full_model_by_dataset`` a dashed line marks
    our model's mean metric, so a factor reaching the line carries most of the
    separating power.

    Args:
        contributions_by_dataset: ``{dataset: {factor: {"f1": ..., "auc": ...}}}``.
        full_model_by_dataset: Optional ``{dataset: {"f1": ..., "auc": ...}}`` for our
            model, averaged for the dashed reference line.
        save_dir: Directory to write the figure into.
        exclude: Dataset names to drop from the average (e.g. the out-of-domain
            balloon); they stay in the per-dataset CSV but are not pooled here.

    Returns:
        Path: The written ``factor_contributions.png``.
    """
    contributions_by_dataset = {k: v for k, v in contributions_by_dataset.items() if k not in exclude}
    if full_model_by_dataset:
        full_model_by_dataset = {k: v for k, v in full_model_by_dataset.items() if k not in exclude}
    factors = [f for f in FACTOR_ORDER if any(f in contrib for contrib in contributions_by_dataset.values())]
    x = np.arange(len(factors))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (key, label) in zip(axes, (("f1", "F1"), ("auc", "AUC")), strict=True):
        stats = [
            _mean_std([c[f][key] for c in contributions_by_dataset.values() if f in c and key in c[f]])
            for f in factors
        ]
        ax.bar(x, [m for m, _ in stats], width=0.6, yerr=[s for _, s in stats], capsize=4, color=COLOR_MODEL)
        if full_model_by_dataset:
            full_mean, _ = _mean_std([fm.get(key, float("nan")) for fm in full_model_by_dataset.values()])
            ax.axhline(full_mean, ls="--", lw=1.5, color=COLOR_REFERENCE, label="Our model (average)")
            ax.legend(loc="lower right", fontsize=LEGEND_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([FACTOR_LABELS[f] for f in factors])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Factor", fontsize=LABEL_SIZE)
        ax.set_ylabel("Score", fontsize=LABEL_SIZE)
        ax.set_title(label, fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

    fig.suptitle("Factor contributions on ≥3-hit events (averaged across datasets)", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return _save(fig, Path(save_dir) / "factor_contributions.png")
