from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

CMAP_SIGNAL = "mako_r"
CMAP_BACKGROUND = "rocket_r"
COLOR_SIGNAL = "#3b6e8c"
COLOR_BACKGROUND = "#b8413a"
COLOR_ROC = "#4c6ef5"
COLOR_PR = "#e8843c"
COLOR_REFERENCE = "#9aa0a6"

SUPTITLE_SIZE = 18
TITLE_SIZE = 17
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
ANNOT_SIZE = 16

# Display metadata for the model features
FEATURE_AXES = {
    "delta_E": {"label": "ΔE [keV]", "name": "ΔE", "scale": "log"},
    "arm": {"label": r"$\Omega$", "name": r"$\Omega$", "scale": "linear"},
    "anni": {"label": r"$\Psi$", "name": r"$\Psi$", "scale": "linear"},
}


def to_numpy(values: object) -> np.ndarray:
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


def axis_meta(feat: str) -> dict[str, str]:
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


def plot_confusion_matrix(tp: int, fp: int, fn: int, tn: int) -> Figure:
    """Plot and save a confusion matrix heatmap.

    Args:
        tp (int): Number of true positive events.
        fp (int): Number of false positive events.
        fn (int): Number of false negative events.
        tn (int): Number of true negative events.

    Returns:
        Figure: The confusion-matrix figure.
    """
    cm = np.array([[tp, fn], [fp, tn]])
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
    scores = to_numpy(scores).ravel()
    labels = to_numpy(labels).ravel() > 0.5

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
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


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
    scores = to_numpy(scores).ravel()
    labels = to_numpy(labels).ravel() > 0.5

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
    labels_np = to_numpy(labels).ravel() > 0.5
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


def plot_sky_backprojection(
    panels: Sequence[tuple[str, torch.Tensor, int]],
    phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    theta_range: tuple[float, float] = (0.0, np.pi),
    title: str = "Compton-cone backprojection",
) -> Figure:
    """Plot backprojected sky images side by side, one panel per event selection.

    Each panel renders one accumulated far-field backprojection (the stacked
    Compton-cone circles of its event selection) as a θ/φ heatmap. Because the
    selections differ by orders of magnitude in event count, every panel is
    stretched independently — percentile-clipped so the diffuse cone-crossing
    background does not swallow the colour range — and absolute pixel values are
    meaningless across panels; each panel instead reports its event count and the
    significance of its brightest pixel, ``(max - mean) / std``. A crosshair at
    the first panel's peak is drawn on all panels, so the eye can compare where
    (and whether) each selection concentrates at the same source position.

    Args:
        panels: One ``(title, image, n_events)`` per panel — the panel title, the
            accumulated ``(n_theta, n_phi)`` sky image, and its event count. The
            first panel is the reference whose peak sets the crosshair.
        phi_range: ``(min, max)`` azimuthal extent of the images, in radians.
        theta_range: ``(min, max)`` polar extent of the images, in radians.
        title: Figure suptitle.

    Returns:
        Figure: The multi-panel sky-image figure.

    Raises:
        ValueError: If ``panels`` is empty.
    """
    if not panels:
        raise ValueError("plot_sky_backprojection needs at least one panel.")

    images = [to_numpy(image) for _, image, _ in panels]
    n_theta, n_phi = images[0].shape
    phi_edges = np.degrees(np.linspace(phi_range[0], phi_range[1], n_phi + 1))
    theta_edges = np.degrees(np.linspace(theta_range[0], theta_range[1], n_theta + 1))

    # Crosshair: the reference (first) panel's brightest pixel, at its bin center.
    peak_theta_idx, peak_phi_idx = np.unravel_index(np.argmax(images[0]), images[0].shape)
    peak_phi = (phi_edges[peak_phi_idx] + phi_edges[peak_phi_idx + 1]) / 2
    peak_theta = (theta_edges[peak_theta_idx] + theta_edges[peak_theta_idx + 1]) / 2

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), sharey=True)
    for ax, (panel_title, _, n_events), image in zip(np.atleast_1d(axes), panels, images, strict=True):
        vmin, vmax = np.percentile(image, [1.0, 99.5])
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.pcolormesh(phi_edges, theta_edges, image, cmap=CMAP_SIGNAL, vmin=vmin, vmax=vmax)
        # A white under-stroke keeps the crosshair legible over the cone texture.
        stroke = [patheffects.withStroke(linewidth=2.5, foreground="white")]
        ax.axvline(peak_phi, color=COLOR_REFERENCE, lw=1, ls="--", path_effects=stroke)
        ax.axhline(peak_theta, color=COLOR_REFERENCE, lw=1, ls="--", path_effects=stroke)

        std = image.std()
        annotation = f"N = {n_events:,}"
        if std > 0:
            annotation += f"\npeak {(image.max() - image.mean()) / std:.1f}σ"
        ax.text(
            0.98,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LEGEND_SIZE,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_title(panel_title, fontsize=TITLE_SIZE)
        ax.set_xlabel("φ [deg]", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

    first = np.atleast_1d(axes)[0]
    first.set_ylabel("θ [deg]", fontsize=LABEL_SIZE)
    first.invert_yaxis()

    fig.suptitle(title, fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return fig


def draw_density_panel(
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


def draw_density_1d_panel(
    ax: plt.Axes,
    values: np.ndarray,
    x_bins: np.ndarray,
    x_meta: dict[str, str],
    title: str,
    cmap: str,
    title_color: str,
) -> None:
    """Draw one class's 1D probability density into ``ax`` (in place).

    The 1D analogue of :func:`draw_density_panel`: bar heights give the
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


def plot_density_1d(term: dict, signal_cmap: str, bg_cmap: str) -> Figure:
    """Plot a fitted 1D density term as stacked β⁺ decay and background panels.

    The 1D analogue of :func:`plot_density_2d`: same two-panel layout, titles, and
    colour language, with one axis instead of two. The panels are stacked
    vertically so the two classes share a column and their probability axes line
    up, making the shift between them read directly.

    Args:
        term: A 1D density term dict (``kind == "1d"``).
        signal_cmap: Colormap for the β⁺ decay (signal) panel.
        bg_cmap: Colormap for the background panel.

    Returns:
        Figure: The two-panel density figure.
    """
    x_meta = axis_meta(term["xfeat"])
    x_bins = to_numpy(term["x_bins"])
    beta, bg = to_numpy(term["beta"]).ravel(), to_numpy(term["bg"]).ravel()

    fig, (ax_beta, ax_bg) = plt.subplots(2, 1, figsize=(7.5, 9))
    joint = f"P({x_meta['name']})"
    draw_density_1d_panel(
        ax_beta, beta, x_bins, x_meta, rf"$\beta^+$ decay: {joint}", signal_cmap, COLOR_SIGNAL
    )
    draw_density_1d_panel(ax_bg, bg, x_bins, x_meta, f"Background: {joint}", bg_cmap, COLOR_BACKGROUND)

    fig.suptitle(f"{joint}  (n_bins={beta.shape[0]})", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    return fig


def plot_density_2d(term: dict, signal_cmap: str, bg_cmap: str) -> Figure:
    """Plot a fitted 2D density term as stacked β⁺ decay and background panels.

    The panels are stacked vertically so both classes share a column and their
    x-axes align, making a feature at a given energy directly comparable between
    them.

    Args:
        term: A 2D density term dict (``kind == "2d"``).
        signal_cmap: Colormap for the β⁺ decay (signal) panel.
        bg_cmap: Colormap for the background panel.

    Returns:
        Figure: The two-panel density figure.
    """
    x_meta, y_meta = axis_meta(term["xfeat"]), axis_meta(term["yfeat"])
    x_bins, y_bins = to_numpy(term["x_bins"]), to_numpy(term["y_bins"])
    beta, bg = to_numpy(term["beta"]), to_numpy(term["bg"])

    fig, (ax_beta, ax_bg) = plt.subplots(2, 1, figsize=(7.5, 10))
    joint = f"P({x_meta['name']}, {y_meta['name']})"
    draw_density_panel(
        ax_beta, beta, x_bins, y_bins, x_meta, y_meta, rf"$\beta^+$ decay: {joint}", signal_cmap, COLOR_SIGNAL
    )
    draw_density_panel(
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
    """Plot a fitted density term, comparing the β⁺ decay and background classes.

    Consumes a density term from a fitted model (``pipeline.train.Trainer``
    ``models``, or a loaded ``.pt`` artifact) and dispatches on its kind: a 1D
    term (``{"kind": "1d", "xfeat", "x_bins", "beta", "bg"}``) is drawn as
    stacked per-class bar panels, and a 2D term (``{"kind": "2d", "xfeat",
    "yfeat", "x_bins", "y_bins", "beta", "bg"}``) as stacked log-scaled
    heatmaps. Both classes share their bin edges, so the comparison is direct.

    Args:
        term: A 1D or 2D density term dict from a fitted model.
        signal_cmap: Colormap for the β⁺ decay (signal) class.
        bg_cmap: Colormap for the background class.

    Returns:
        Figure: The two-panel (β⁺ decay vs background) density figure.

    Raises:
        ValueError: If ``term`` is neither a 1D nor a 2D density term.
    """
    kind = term.get("kind")
    if kind == "1d":
        return plot_density_1d(term, signal_cmap, bg_cmap)
    if kind == "2d":
        return plot_density_2d(term, signal_cmap, bg_cmap)
    raise ValueError(f"plot_density_matrix needs a 1D or 2D term, got kind={kind!r}.")
