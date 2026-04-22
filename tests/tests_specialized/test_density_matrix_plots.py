"""
Density matrix visualization at multiple bin resolutions.

For each pair of features, renders side-by-side (signal | background)
probability heatmaps at BIN_SIZES = [20, 300, 1000]:

    1) P(ΔE, ARM)
    2) P(ΔE, annihilation angle)
    3) P(ARM, annihilation angle)

Bin edges are derived from the combined (signal + background) feature
tensors so the per-class matrices share the same coordinate system. ΔE and
ARM are residual-like (peak near 0, long tail) and use log spacing;
annihilation angle is a bounded cosine and uses linear spacing.

No classifier is built here — these are the raw density-matrix diagnostics
that explain how class separation evolves with resolution before the
curse of dimensionality kicks in at 1000×1000.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
import torch
import wandb
from modeling.matrix_calculations import build_density_matrix

BIN_SIZES = [20, 300, 1000]

_LABEL_DE = "ΔE [keV]"
_LABEL_ARM = "ARM [rad]"
_LABEL_ANGLE = "Annihilation angle [cosine]"


def _to_np(t: torch.Tensor):
    return t.detach().cpu().float().numpy()


def _plot_heatmap(
    ax,
    matrix: torch.Tensor,
    x_bins: torch.Tensor,
    y_bins: torch.Tensor,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str,
    log_x: bool,
    log_y: bool,
) -> None:
    """Draw a 2D probability heatmap on a log colour scale.

    Uses LogNorm because probability matrices are extremely sparse at the
    high-resolution end: a handful of peak cells carry most of the mass and a
    linear colourmap flattens everything else to zero. ``rasterized=True``
    collapses the ~n_bins² quad patches into a single raster image at render
    time, which keeps 1000×1000 plots fast and well under memory limits.
    """
    data = _to_np(matrix).T  # pcolormesh expects (n_y, n_x)
    xb = _to_np(x_bins)
    yb = _to_np(y_bins)

    vmin = data[data > 0].min() if (data > 0).any() else 1e-10
    norm = mcolors.LogNorm(vmin=vmin, vmax=data.max())

    im = ax.pcolormesh(
        xb,
        yb,
        data,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        linewidth=0,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    plt.colorbar(im, ax=ax, label="Probability")


def _build_pair(
    combined_x: torch.Tensor,
    combined_y: torch.Tensor,
    bdecay_x: torch.Tensor,
    bdecay_y: torch.Tensor,
    bg_x: torch.Tensor,
    bg_y: torch.Tensor,
    n_bins: int,
    spacing_x: str,
    spacing_y: str,
    log_x_floor: float | None = None,
    log_y_floor: float | None = None,
):
    """Shared-coordinate build: derive edges once from the combined tensor,
    then reuse them for signal and background so the two matrices sit in the
    same bin system and can be visually compared cell-by-cell."""
    kwargs = {"n_bins_x": n_bins, "n_bins_y": n_bins, "spacing_x": spacing_x, "spacing_y": spacing_y}
    if log_x_floor is not None:
        kwargs["log_x_floor"] = log_x_floor
    if log_y_floor is not None:
        kwargs["log_y_floor"] = log_y_floor

    _, x_bins, y_bins = build_density_matrix(combined_x, combined_y, **kwargs)
    bdecay_probs, _, _ = build_density_matrix(bdecay_x, bdecay_y, x_bins=x_bins, y_bins=y_bins)
    bg_probs, _, _ = build_density_matrix(bg_x, bg_y, x_bins=x_bins, y_bins=y_bins)
    return bdecay_probs, bg_probs, x_bins, y_bins


@pytest.fixture(scope="module", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def pair_matrices(real_tensors, request):
    """Build all three pairwise density matrices (signal + background) at the
    requested resolution. Module-scoped so the three plot tests at the same
    n_bins reuse the build instead of recomputing six matrices each."""
    n_bins = request.param
    t = real_tensors

    # ΔE vs ARM — both log-spaced residual-like axes
    de_arm_bdecay, de_arm_bg, x_de_arm, y_de_arm = _build_pair(
        t["combined_dE"],
        t["combined_arm"],
        t["bdecay_dE"],
        t["bdecay_arm"],
        t["bg_dE"],
        t["bg_arm"],
        n_bins=n_bins,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=0.1,
        log_y_floor=1e-3,
    )

    # ΔE vs annihilation angle — log × linear (cosine is bounded)
    de_angle_bdecay, de_angle_bg, x_de_angle, y_de_angle = _build_pair(
        t["combined_dE"],
        t["combined_angle"],
        t["bdecay_dE"],
        t["bdecay_angle"],
        t["bg_dE"],
        t["bg_angle"],
        n_bins=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )

    # ARM vs annihilation angle — log × linear
    arm_angle_bdecay, arm_angle_bg, x_arm_angle, y_arm_angle = _build_pair(
        t["combined_arm"],
        t["combined_angle"],
        t["bdecay_arm"],
        t["bdecay_angle"],
        t["bg_arm"],
        t["bg_angle"],
        n_bins=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=1e-3,
    )

    return {
        "n_bins": n_bins,
        "de_arm": {
            "bdecay": de_arm_bdecay,
            "bg": de_arm_bg,
            "x_bins": x_de_arm,
            "y_bins": y_de_arm,
        },
        "de_angle": {
            "bdecay": de_angle_bdecay,
            "bg": de_angle_bg,
            "x_bins": x_de_angle,
            "y_bins": y_de_angle,
        },
        "arm_angle": {
            "bdecay": arm_angle_bdecay,
            "bg": arm_angle_bg,
            "x_bins": x_arm_angle,
            "y_bins": y_arm_angle,
        },
    }


def _log_pair_plot(
    pair: dict,
    n_bins: int,
    pair_title: str,
    x_label: str,
    y_label: str,
    log_x: bool,
    log_y: bool,
    wandb_key: str,
    wandb_run,
) -> None:
    """Render a 2-panel (β-decay | background) heatmap for one feature pair,
    assert both matrices are proper probability distributions, and log the
    figure + a few scalar summaries to W&B."""
    bdecay = pair["bdecay"]
    bg = pair["bg"]

    assert torch.isclose(bdecay.sum(), torch.tensor(1.0), atol=1e-5), (
        f"{pair_title}: β-decay matrix does not sum to 1 (got {bdecay.sum().item():.6f})"
    )
    assert torch.isclose(bg.sum(), torch.tensor(1.0), atol=1e-5), (
        f"{pair_title}: background matrix does not sum to 1 (got {bg.sum().item():.6f})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_heatmap(
        axes[0],
        bdecay,
        pair["x_bins"],
        pair["y_bins"],
        f"β-decay: {pair_title}",
        x_label,
        y_label,
        cmap="viridis_r",
        log_x=log_x,
        log_y=log_y,
    )
    _plot_heatmap(
        axes[1],
        bg,
        pair["x_bins"],
        pair["y_bins"],
        f"Background: {pair_title}",
        x_label,
        y_label,
        cmap="plasma_r",
        log_x=log_x,
        log_y=log_y,
    )
    fig.suptitle(f"{pair_title}  (n_bins={n_bins})", fontsize=12)
    fig.tight_layout()

    wandb_run.log(
        {
            f"density_matrix/n_bins={n_bins}/{wandb_key}/heatmap": wandb.Image(fig),
            f"density_matrix/n_bins={n_bins}/{wandb_key}/bdecay_peak": float(bdecay.max().item()),
            f"density_matrix/n_bins={n_bins}/{wandb_key}/bg_peak": float(bg.max().item()),
            f"density_matrix/n_bins={n_bins}/{wandb_key}/bdecay_nonzero_frac": float(
                (bdecay > 0).float().mean().item()
            ),
            f"density_matrix/n_bins={n_bins}/{wandb_key}/bg_nonzero_frac": float(
                (bg > 0).float().mean().item()
            ),
        }
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ΔE vs ARM
# --------------------------------------------------------------------------- #


def test_density_matrix_deltaE_arm(pair_matrices, wandb_run):
    """β-decay and background P(ΔE, ARM) heatmaps side-by-side."""
    _log_pair_plot(
        pair_matrices["de_arm"],
        n_bins=pair_matrices["n_bins"],
        pair_title="P(ΔE, ARM)",
        x_label=_LABEL_DE,
        y_label=_LABEL_ARM,
        log_x=True,
        log_y=True,
        wandb_key="deltaE_arm",
        wandb_run=wandb_run,
    )


# --------------------------------------------------------------------------- #
# ΔE vs annihilation angle
# --------------------------------------------------------------------------- #


def test_density_matrix_deltaE_angle(pair_matrices, wandb_run):
    """β-decay and background P(ΔE, annihilation angle) heatmaps side-by-side."""
    _log_pair_plot(
        pair_matrices["de_angle"],
        n_bins=pair_matrices["n_bins"],
        pair_title="P(ΔE, annihilation angle)",
        x_label=_LABEL_DE,
        y_label=_LABEL_ANGLE,
        log_x=True,
        log_y=False,
        wandb_key="deltaE_angle",
        wandb_run=wandb_run,
    )


# --------------------------------------------------------------------------- #
# ARM vs annihilation angle
# --------------------------------------------------------------------------- #


def test_density_matrix_arm_angle(pair_matrices, wandb_run):
    """β-decay and background P(ARM, annihilation angle) heatmaps side-by-side."""
    _log_pair_plot(
        pair_matrices["arm_angle"],
        n_bins=pair_matrices["n_bins"],
        pair_title="P(ARM, annihilation angle)",
        x_label=_LABEL_ARM,
        y_label=_LABEL_ANGLE,
        log_x=True,
        log_y=False,
        wandb_key="arm_angle",
        wandb_run=wandb_run,
    )
