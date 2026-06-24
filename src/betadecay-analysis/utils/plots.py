import matplotlib

# Force the non-interactive Agg backend before pyplot is imported: these figures
# are only ever saved to file or logged to W&B, never shown, and the analysis
# pipeline runs headless (no display) on the production VM. Without this,
# matplotlib may pick a GUI backend and crash on the first figure with a
# "no display"/"couldn't connect to display" error.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

# Single source of truth for plot colours. Every figure in this module pulls
# from here, so the whole look changes from one block. The mapping is *semantic*:
# β-decay (signal) and background each own a perceptually-uniform, colorblind-safe
# seaborn colormap (reversed so higher probability reads darker) and a matching
# categorical accent used for that class's panel title; the metric curves get
# their own accents; and every dashed reference line shares one neutral grey.
CMAP_SIGNAL = "mako_r"  # β-decay density panels (light -> dark teal)
CMAP_BACKGROUND = "rocket_r"  # background density panels (light -> dark red)
COLOR_SIGNAL = "#3b6e8c"  # β-decay accent (mako teal), used for its panel title
COLOR_BACKGROUND = "#b8413a"  # background accent (rocket red), used for its panel title
COLOR_ROC = "#4c6ef5"  # ROC curve (indigo)
COLOR_PR = "#e8843c"  # precision-recall curve (amber)
COLOR_REFERENCE = "#9aa0a6"  # all dashed baselines / chance lines (neutral grey)

# Font sizes for paper-quality figures (kept in sync with ablations/plots.py, well
# above the matplotlib defaults so titles, axes, and legends stay legible in print).
SUPTITLE_SIZE = 18
TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
ANNOT_SIZE = 16

# Display metadata for the model features, keyed by the feature names used in
# dataset.datasets.FEATURES. ``label`` is the axis label (with unit), ``name``
# the short symbol for titles, and ``scale`` the axis scaling that matches how
# the density's bins were built in pipeline.train (delta_E/arm are log-spaced,
# anni is linear). Unknown features fall back to a linear axis labelled by name.
FEATURE_AXES = {
    "delta_E": {"label": "ΔE [keV]", "name": "ΔE", "scale": "log"},
    "arm": {"label": "ARM [rad]", "name": "ARM", "scale": "log"},
    "anni": {"label": "Annihilation score", "name": "Annih.", "scale": "linear"},
}


def _to_numpy(values: object) -> np.ndarray:
    """Convert a torch tensor or array-like to a float NumPy array.

    Keeps this module free of a hard ``torch`` import (and of the ROOT-backed
    pipeline packages) by duck-typing the tensor ``detach``/``cpu`` protocol.

    Args:
        values: A torch tensor, NumPy array, or any array-like of numbers.

    Returns:
        np.ndarray: The values as a ``float`` NumPy array (shape preserved).
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values, dtype=float)


def _axis_meta(feat: str) -> dict[str, str]:
    """Display metadata for a feature, with a linear fallback for unknown names."""
    return FEATURE_AXES.get(feat, {"label": feat, "name": feat, "scale": "linear"})


def density_term_features(term: dict) -> str:
    """Feature slug for a density term, shared by its W&B key and saved filename.

    Single source of truth so the W&B key (``model/density/n<bucket>/<slug>``) and
    the on-disk filename (``density_n<bucket>_<slug>.png``) can never drift apart.

    Args:
        term: A 1D or 2D density term dict.

    Returns:
        str: ``xfeat`` for a 1D term, or ``"<xfeat>_<yfeat>"`` for a 2D term.
    """
    return term["xfeat"] if term.get("kind") == "1d" else f"{term['xfeat']}_{term['yfeat']}"


def plot_confusion_matrix(TP: int, FP: int, FN: int, TN: int) -> Figure:
    """Plot and save a confusion matrix heatmap.

    Args:
        TP (int): Number of true positive events.
        FP (int): Number of false positive events.
        FN (int): Number of false negative events.
        TN (int): Number of true negative events.

    Returns:
        Figure: The confusion-matrix figure.
    """
    cm = np.array([[TP, FN], [FP, TN]])
    labels = ["Positive", "Negative"]

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CMAP_SIGNAL,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": ANNOT_SIZE},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=LABEL_SIZE)
    ax.set_ylabel("Actual", fontsize=LABEL_SIZE)
    ax.set_title("Confusion matrix", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    fig.tight_layout()
    return fig


def roc_curve(scores: object, labels: object) -> tuple[np.ndarray, np.ndarray, float]:
    """Tie-aware ROC curve points and area under the curve.

    Sweeps every distinct score as a decision cut (predicting signal for
    ``scores >= cut``) and records the false- and true-positive rates, prepended
    with the origin. The area is the trapezoidal integral of TPR over FPR, which
    equals the rank-statistic AUC reported by the pipeline.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: ``(fpr, tpr, auc)``. ``auc`` is NaN
        when either class is absent (the curve is then degenerate).
    """
    scores = _to_numpy(scores).ravel()
    labels = _to_numpy(labels).ravel() > 0.5

    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    n_pos, n_neg = tp[-1], fp[-1]
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # Keep only the last index of each tied-score group: a cut between equal
    # scores is not a valid operating point.
    distinct = np.append(np.diff(s) != 0, True)
    tpr = np.concatenate([[0.0], tp[distinct] / n_pos])
    fpr = np.concatenate([[0.0], fp[distinct] / n_neg])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def pr_curve(scores: object, labels: object) -> tuple[np.ndarray, np.ndarray, float]:
    """Tie-aware precision-recall curve points and average precision.

    Sweeps every distinct score as a decision cut and records recall and
    precision, anchored at ``(recall=0, precision=1)``. Average precision is the
    recall-weighted sum of precisions over the curve.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: ``(recall, precision, ap)``. ``ap``
        is NaN when there are no positives.
    """
    scores = _to_numpy(scores).ravel()
    labels = _to_numpy(labels).ravel() > 0.5

    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    n_pos = tp[-1]
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")

    distinct = np.append(np.diff(s) != 0, True)
    recall = np.concatenate([[0.0], tp[distinct] / n_pos])
    precision = np.concatenate([[1.0], tp[distinct] / np.clip(tp[distinct] + fp[distinct], 1.0, None)])
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return recall, precision, ap


def plot_roc_curve(scores: object, labels: object, title: str = "ROC curve") -> Figure:
    """Plot the ROC curve of per-event scores against the chance diagonal.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.
        title: Axes title.

    Returns:
        Figure: The ROC figure, with the AUC shown in the legend.
    """
    fpr, tpr, auc = roc_curve(scores, labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=COLOR_ROC, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=COLOR_REFERENCE, lw=1, ls="--", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("True positive rate", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def plot_pr_curve(scores: object, labels: object, title: str = "Precision-recall curve") -> Figure:
    """Plot the precision-recall curve of per-event scores against the prevalence baseline.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event truth labels; non-zero is the positive (β⁺) class.
        title: Axes title.

    Returns:
        Figure: The precision-recall figure, with the average precision in the legend.
    """
    recall, precision, ap = pr_curve(scores, labels)
    labels_np = _to_numpy(labels).ravel() > 0.5
    prevalence = float(labels_np.mean()) if labels_np.size else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color=COLOR_PR, lw=2, label=f"AP = {ap:.4f}")
    ax.axhline(prevalence, color=COLOR_REFERENCE, lw=1, ls="--", label=f"Baseline = {prevalence:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall", fontsize=LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def _draw_density_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    x_meta: dict[str, str],
    y_meta: dict[str, str],
    title: str,
    cmap: str,
    title_color: str,
) -> None:
    """Draw one class's 2D probability density into ``ax`` (in place).

    Renders the density over its (possibly non-uniform, log-spaced) bin edges
    with a log-normalized colour scale; cells with zero probability are masked so
    they read as empty rather than as the colormap floor.

    Args:
        ax: Target axes.
        matrix: ``(n_x, n_y)`` probability matrix indexed ``[x_idx, y_idx]``.
        x_bins: ``n_x + 1`` x-axis bin edges.
        y_bins: ``n_y + 1`` y-axis bin edges.
        x_meta: Display metadata (``label``/``scale``) for the x feature.
        y_meta: Display metadata (``label``/``scale``) for the y feature.
        title: Panel title.
        cmap: Matplotlib colormap name.
        title_color: Colour for the panel title (the class's categorical accent).
    """
    # pcolormesh expects C as (n_y, n_x) against the edge grids, so transpose the
    # [x_idx, y_idx] matrix; mask non-positive cells out of the LogNorm.
    grid = np.ma.masked_less_equal(matrix.T, 0.0)
    positive = matrix[matrix > 0]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None

    mesh = ax.pcolormesh(x_bins, y_bins, grid, cmap=cmap, norm=norm)
    ax.set_xscale(x_meta["scale"])
    ax.set_yscale(y_meta["scale"])
    ax.set_xlabel(x_meta["label"], fontsize=LABEL_SIZE)
    ax.set_ylabel(y_meta["label"], fontsize=LABEL_SIZE)
    ax.set_title(title, color=title_color, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label("Probability", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)


def _draw_density_1d_panel(
    ax: plt.Axes,
    values: np.ndarray,
    x_bins: np.ndarray,
    x_meta: dict[str, str],
    title: str,
    cmap: str,
    title_color: str,
) -> None:
    """Draw one class's 1D probability density into ``ax`` (in place).

    The 1D analogue of :func:`_draw_density_panel`: bar heights give the
    probability per (possibly log-spaced) bin and the bars are shaded by that
    same probability through a log-normalized colour scale, so the panel matches
    the 2D heatmap's colour language. Zero-probability bins are masked out of the
    scale (and render as empty zero-height bars).

    Args:
        ax: Target axes.
        values: ``(n_bins,)`` probability vector.
        x_bins: ``n_bins + 1`` x-axis bin edges.
        x_meta: Display metadata (``label``/``scale``) for the x feature.
        title: Panel title.
        cmap: Matplotlib colormap name.
        title_color: Colour for the panel title (the class's categorical accent).
    """
    positive = values[values > 0]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None
    colormap = plt.get_cmap(cmap)
    shading = np.ma.masked_less_equal(values, 0.0)
    colors = colormap(norm(shading)) if norm is not None else colormap(np.zeros_like(values))

    ax.bar(x_bins[:-1], values, width=np.diff(x_bins), align="edge", color=colors)
    ax.set_xscale(x_meta["scale"])
    ax.set_xlabel(x_meta["label"], fontsize=LABEL_SIZE)
    ax.set_ylabel("Probability", fontsize=LABEL_SIZE)
    ax.set_title(title, color=title_color, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    if norm is not None:
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        cbar.set_label("Probability", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)


def _plot_density_1d(term: dict, signal_cmap: str, bg_cmap: str) -> Figure:
    """Plot a fitted 1D density term as side-by-side β-decay and background panels.

    The 1D analogue of :func:`_plot_density_2d`: same two-panel layout, titles, and
    colour language, with one axis instead of two.

    Args:
        term: A 1D density term dict (``kind == "1d"``).
        signal_cmap: Colormap for the β-decay (signal) panel.
        bg_cmap: Colormap for the background panel.

    Returns:
        Figure: The two-panel density figure.
    """
    x_meta = _axis_meta(term["xfeat"])
    x_bins = _to_numpy(term["x_bins"])
    beta, bg = _to_numpy(term["beta"]).ravel(), _to_numpy(term["bg"]).ravel()

    fig, (ax_beta, ax_bg) = plt.subplots(1, 2, figsize=(14, 5))
    joint = f"P({x_meta['name']})"
    _draw_density_1d_panel(ax_beta, beta, x_bins, x_meta, f"β-decay: {joint}", signal_cmap, COLOR_SIGNAL)
    _draw_density_1d_panel(ax_bg, bg, x_bins, x_meta, f"Background: {joint}", bg_cmap, COLOR_BACKGROUND)

    fig.suptitle(f"{joint}  (n_bins={beta.shape[0]})", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return fig


def _plot_density_2d(term: dict, signal_cmap: str, bg_cmap: str) -> Figure:
    """Plot a fitted 2D density term as side-by-side β-decay and background panels.

    Args:
        term: A 2D density term dict (``kind == "2d"``).
        signal_cmap: Colormap for the β-decay (signal) panel.
        bg_cmap: Colormap for the background panel.

    Returns:
        Figure: The two-panel density figure.
    """
    x_meta, y_meta = _axis_meta(term["xfeat"]), _axis_meta(term["yfeat"])
    x_bins, y_bins = _to_numpy(term["x_bins"]), _to_numpy(term["y_bins"])
    beta, bg = _to_numpy(term["beta"]), _to_numpy(term["bg"])

    fig, (ax_beta, ax_bg) = plt.subplots(1, 2, figsize=(14, 5))
    joint = f"P({x_meta['name']}, {y_meta['name']})"
    _draw_density_panel(
        ax_beta, beta, x_bins, y_bins, x_meta, y_meta, f"β-decay: {joint}", signal_cmap, COLOR_SIGNAL
    )
    _draw_density_panel(
        ax_bg, bg, x_bins, y_bins, x_meta, y_meta, f"Background: {joint}", bg_cmap, COLOR_BACKGROUND
    )

    # Report the actual per-axis bin counts (the x- and y-axes are sized
    # independently), collapsing to a single number only when the grid is square.
    n_x, n_y = beta.shape
    n_bins = f"{n_x}" if n_x == n_y else f"{n_x}×{n_y}"
    fig.suptitle(f"{joint}  (n_bins={n_bins})", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return fig


def plot_density_matrix(
    term: dict,
    signal_cmap: str = CMAP_SIGNAL,
    bg_cmap: str = CMAP_BACKGROUND,
) -> Figure:
    """Plot a fitted density term, comparing the β-decay and background classes.

    Consumes a density term from a fitted model (``pipeline.train.Trainer``
    ``models``, or a loaded ``.pt`` artifact) and dispatches on its kind: a 1D
    term (``{"kind": "1d", "xfeat", "x_bins", "beta", "bg"}``) is drawn as overlaid
    step densities on one axes, and a 2D term (``{"kind": "2d", "xfeat", "yfeat",
    "x_bins", "y_bins", "beta", "bg"}``) as side-by-side log-scaled heatmaps. Both
    classes share their bin edges, so the comparison is direct.

    Args:
        term: A 1D or 2D density term dict from a fitted model.
        signal_cmap: Colormap for the β-decay (signal) class.
        bg_cmap: Colormap for the background class.

    Returns:
        Figure: The density figure (one axes for 1D, two panels for 2D).

    Raises:
        ValueError: If ``term`` is neither a 1D nor a 2D density term.
    """
    kind = term.get("kind")
    if kind == "1d":
        return _plot_density_1d(term, signal_cmap, bg_cmap)
    if kind == "2d":
        return _plot_density_2d(term, signal_cmap, bg_cmap)
    raise ValueError(f"plot_density_matrix needs a 1D or 2D term, got kind={kind!r}.")
