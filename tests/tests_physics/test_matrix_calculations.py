"""
Unit tests for physics/matrix_calculations.py.

Covers:
    - build_density_matrix   (11 tests — 8 edge-case, 3 real-data)
    - lookup_density_values  ( 7 tests — 6 edge-case, 1 real-data)
    - Physics tests          (14 tests — TRUE vs FALSE event matrix analysis)

Test strategy
-------------
1. Edge-case / contract tests  (inline tensors)
   Require specific, controlled inputs to verify mathematical guarantees:
   NaN/inf filtering, ValueError on empty input, exact probability values,
   boundary conditions, and clamping behaviour.

2. Real-data structural tests  (real_tensors fixture)
   Verify correct behaviour on Activation.sim feature distributions.

3. Physics tests  (real_tensors + density_matrices + wandb_run fixtures)
   Assert that the density matrices are physically meaningful by checking
   how they behave for TRUE (beta-decay) vs FALSE (background) events.
   Every test logs one diagnostic plot to Weights & Biases.

   Test groups:
     - TRUE events structure  (2 tests)  — mass concentrated at low delta_E / ARM
     - FALSE events structure (2 tests)  — mass more spread, not peaked near zero
     - Entropy comparison     (2 tests)  — signal always less entropic than background
     - Distinguishability     (4 tests)  — total variation + log-ratio sign checks
     - Ground-truth accuracy  (4 tests)  — event-level density ratio on gen data
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb

from physics.matrix_calculations import build_density_matrix, lookup_density_values

# ===========================================================================
# Module-level plot helpers
# ===========================================================================

# Axis labels reused across multiple tests
_LABEL_DE = "delta_E [keV]"
_LABEL_ANGLE = "Annihilation angle [cosine]"
_LABEL_ARM = "ARM [rad]"


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Detach and convert a tensor to a float32 numpy array."""
    return t.detach().cpu().float().numpy()


def _entropy(matrix: torch.Tensor) -> float:
    """Shannon entropy (nats) of a probability matrix. Zero cells are skipped."""
    p = matrix.flatten()
    p = p[p > 0]
    return -(p * torch.log(p)).sum().item()


def _total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total variation distance in [0, 1] between two probability distributions."""
    return 0.5 * torch.abs(p - q).sum().item()


def _plot_heatmap(
    ax,
    matrix: torch.Tensor,
    x_bins: torch.Tensor,
    y_bins: torch.Tensor,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str = "viridis",
) -> None:
    """Draw a 2D probability heatmap using pcolormesh with a log colour scale.

    The axes show normal numbers.  Only the colourbar spacing is logarithmic,
    which is necessary because probability matrices are extremely sparse: a few
    cells near the signal peak hold almost all the probability mass, while the
    vast majority are near-zero.  A linear colour scale would map everything
    except the peak to the bottom of the colourmap, making the plot appear flat
    and featureless.  LogNorm preserves contrast across the full dynamic range
    while keeping every label on the axes and colourbar as a plain number.
    """
    data = _to_np(matrix).T          # pcolormesh expects (n_y, n_x)
    xb   = _to_np(x_bins)
    yb   = _to_np(y_bins)

    vmin = data[data > 0].min() if (data > 0).any() else 1e-10
    norm = mcolors.LogNorm(vmin=vmin, vmax=data.max())

    im = ax.pcolormesh(xb, yb, data, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.colorbar(im, ax=ax, label="Probability density")


def _plot_1d_overlay(
    ax,
    bdecay_m: np.ndarray,
    bg_m: np.ndarray,
    bin_centers: np.ndarray,
    x_label: str,
    title: str,
) -> None:
    """Draw overlaid signal and background 1D marginal distributions."""
    ax.plot(bin_centers, bdecay_m, color="steelblue", lw=2, label="Beta-decay")
    ax.plot(bin_centers, bg_m, color="firebrick", lw=2, label="Background")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability")
    ax.set_title(title, fontsize=10)
    ax.legend()


def _bin_centers(edges: torch.Tensor) -> np.ndarray:
    """Return midpoints of histogram bin edges."""
    e = _to_np(edges)
    return 0.5 * (e[:-1] + e[1:])


# ===========================================================================
# build_density_matrix — edge-case / contract tests
# ===========================================================================


def test_nan_values_in_x_are_filtered():
    """
    Pairs where x is NaN must be silently excluded before binning.

    The remaining valid pairs should still produce a well-formed probability
    matrix that sums to 1.  The function must not raise or propagate NaN.
    """
    x = torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    probs, _, _ = build_density_matrix(x, y, n_bins_x=5, n_bins_y=5)

    assert not torch.any(torch.isnan(probs)), "Matrix contains NaN after filtering"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_inf_values_are_filtered():
    """
    Pairs where either x or y is ±inf must be excluded before binning.

    In this test, two of the five pairs are invalid (one +inf in x, one
    -inf in y).  The three remaining valid pairs must form a valid
    probability matrix.
    """
    x = torch.tensor([1.0, 2.0, float("inf"), 4.0, 5.0])
    y = torch.tensor([1.0, 2.0, 3.0, float("-inf"), 5.0])

    probs, _, _ = build_density_matrix(x, y, n_bins_x=5, n_bins_y=5)

    assert not torch.any(torch.isnan(probs)), "Matrix contains NaN after filtering"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_all_nan_raises_value_error():
    """
    If every element in x (or y) is NaN, no valid pairs remain after
    filtering.  build_density_matrix must raise a ValueError with a
    descriptive message rather than silently returning an empty matrix.
    """
    x = torch.full((10,), float("nan"))
    y = torch.full((10,), float("nan"))

    with pytest.raises(ValueError, match="Cannot create probability matrix from empty tensors"):
        build_density_matrix(x, y)


def test_empty_tensor_raises_value_error():
    """
    Passing zero-length tensors must raise a ValueError.

    There are no data points to histogram, so building a probability matrix
    is undefined.
    """
    x = torch.tensor([], dtype=torch.float32)
    y = torch.tensor([], dtype=torch.float32)

    with pytest.raises(ValueError, match="Cannot create probability matrix from empty tensors"):
        build_density_matrix(x, y)


def test_degenerate_range_no_crash():
    """
    When all x values are identical (x_min == x_max), the function guards
    against a zero-width range by adding 1e-6 to x_max.

    The call must not raise, the returned x_bins must have the correct
    number of edges, and the probability matrix must still sum to 1.
    """
    x = torch.full((50,), 3.14)
    y = torch.linspace(0.0, 1.0, 50)

    probs, x_bins, _ = build_density_matrix(x, y, n_bins_x=10, n_bins_y=10)

    assert x_bins.numel() == 11, "Expected 11 x-bin edges for n_bins_x=10"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_custom_bins_are_used_and_returned_unchanged():
    """
    When explicit x_bins and y_bins are supplied, the returned bin tensors
    must be identical to the ones provided.  The matrix shape must match
    (len(x_bins)-1, len(y_bins)-1).

    This guarantees that a second matrix built with the same bin edges is
    in the exact same coordinate system as the first.
    """
    x = torch.linspace(0.0, 10.0, 100)
    y = torch.linspace(0.0, 10.0, 100)
    x_bins = torch.linspace(0.0, 10.0, 21)  # 20 bins
    y_bins = torch.linspace(0.0, 10.0, 16)  # 15 bins

    probs, ret_x_bins, ret_y_bins = build_density_matrix(x, y, x_bins=x_bins, y_bins=y_bins)

    assert torch.allclose(ret_x_bins, x_bins), "Returned x_bins differ from supplied x_bins"
    assert torch.allclose(ret_y_bins, y_bins), "Returned y_bins differ from supplied y_bins"
    assert probs.shape == (20, 15)


def test_points_on_lower_bin_edge_are_counted_and_round_trip_correctly():
    """
    Samples that land exactly on the minimum supplied bin edge must be counted
    into the first cell, not dropped during matrix construction.

    This is a high-value contract test because lookup_density_values maps an
    exact-edge query back into the first bin.  If build_density_matrix were to
    discard the same training point, training and inference would disagree on
    the coordinate system at the boundary.
    """
    x_bins = torch.tensor([0.0, 0.5, 1.0])
    y_bins = torch.tensor([0.0, 0.5, 1.0])

    x = torch.tensor([0.0, 0.0, 0.0])
    y = torch.tensor([0.0, 0.0, 0.0])

    probs, _, _ = build_density_matrix(x, y, x_bins=x_bins, y_bins=y_bins)
    looked_up = lookup_density_values(torch.tensor([0.0]), torch.tensor([0.0]), probs, x_bins, y_bins)

    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(probs[0, 0], torch.tensor(1.0), atol=1e-6), (
        f"Expected all mass in first cell, got probs[0,0] = {probs[0, 0].item():.6f}"
    )
    assert torch.isclose(looked_up[0], torch.tensor(1.0), atol=1e-6), (
        f"Expected lookup at exact lower edge to return 1.0, got {looked_up[0].item():.6f}"
    )


def test_concentrated_data_fills_single_cell():
    """
    When all data points are tightly clustered in one location, virtually
    all probability mass should land in a single cell.

    The maximum cell probability must exceed 0.9 — otherwise the binning
    is spreading mass across multiple cells incorrectly.
    """
    torch.manual_seed(1)
    x = torch.full((200,), 5.0) + torch.randn(200) * 1e-4
    y = torch.full((200,), 5.0) + torch.randn(200) * 1e-4

    probs, _, _ = build_density_matrix(x, y, n_bins_x=5, n_bins_y=5)

    assert probs.max().item() > 0.9, f"Expected max cell probability > 0.9, got {probs.max().item():.4f}"


# ===========================================================================
# build_density_matrix — real-data structural tests
# ===========================================================================


def test_probabilities_sum_to_one_on_real_data(real_tensors):
    """
    The probability matrix built from real beta-decay feature vectors must
    sum to exactly 1.0.

    This is the most fundamental contract: dividing raw counts by their total
    always produces a proper probability distribution, regardless of the
    input distribution.
    """
    probs, _, _ = build_density_matrix(
        real_tensors["bdecay_tensor_delta_E"],
        real_tensors["bdecay_tensor_annihilation_angle"],
    )

    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), (
        f"Expected probability sum = 1.0, got {probs.sum().item():.8f}"
    )


def test_output_shape_with_real_combined_data(real_tensors):
    """
    The probability matrix built from the combined (bdecay + background)
    real feature tensors must have shape (n_bins_x, n_bins_y), with bin-edge
    tensors of lengths n_bins_x+1 and n_bins_y+1.
    """
    n_bins_x, n_bins_y = 20, 30

    probs, x_bins, y_bins = build_density_matrix(
        real_tensors["combined_tensor_delta_E"],
        real_tensors["combined_tensor_annihilation_angle"],
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
    )

    assert probs.shape == (n_bins_x, n_bins_y), f"Expected ({n_bins_x}, {n_bins_y}), got {tuple(probs.shape)}"
    assert x_bins.numel() == n_bins_x + 1
    assert y_bins.numel() == n_bins_y + 1


def test_sized_feature_functions_ignore_nan_padding():
    """
    Masked feature functions must ignore NaN-padded hits rather than treating
    them as physical interactions.
    """
    cfg = OmegaConf.create({"likelihoods": {"arm": {"sigma_x": 4.0, "n_sigma_cos_theta_kin": 1}}})

    energies = torch.tensor(
        [
            [200.0, 311.0, float("nan")],
            [200.0, 100.0, 211.0],
        ],
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    sizes = torch.tensor([2, 3])

    masked_delta_e = delta_E(energies, sizes=sizes)
    masked_angle = annihilation_angle(positions, sizes=sizes)
    masked_arm = arm(energies, positions, cfg, sizes=sizes)

    assert torch.isclose(masked_delta_e, torch.tensor([0.0])).all()
    assert torch.isfinite(masked_angle).all(), f"Expected finite masked annihilation angle, got {masked_angle}"
    assert torch.isfinite(masked_arm).all(), f"Expected finite masked ARM, got {masked_arm}"


def test_shared_bins_produce_consistent_shapes_on_real_data(real_tensors):
    """
    Bins derived from the combined tensor and re-used for signal/background
    matrices must yield identical shapes across all three matrices — proving
    that the shared-coordinate-system contract holds on real detector data.
    """
    _, x_bins, y_bins = build_density_matrix(
        real_tensors["combined_tensor_delta_E"],
        real_tensors["combined_tensor_annihilation_angle"],
    )
    expected = (x_bins.numel() - 1, y_bins.numel() - 1)

    bdecay_probs, _, _ = build_density_matrix(
        real_tensors["bdecay_tensor_delta_E"],
        real_tensors["bdecay_tensor_annihilation_angle"],
        x_bins=x_bins,
        y_bins=y_bins,
    )
    bg_probs, _, _ = build_density_matrix(
        real_tensors["bg_tensor_delta_E"],
        real_tensors["bg_tensor_annihilation_angle"],
        x_bins=x_bins,
        y_bins=y_bins,
    )

    assert bdecay_probs.shape == expected
    assert bg_probs.shape == expected


# ===========================================================================
# lookup_density_values — edge-case / contract tests
# ===========================================================================


def test_known_point_returns_correct_probability():
    """
    Round-trip accuracy: build a 2×2 probability matrix from two points that
    fall into distinct, predictable cells, then verify that a lookup at each
    point returns the exact stored probability.

    Matrix layout:
        x_bins = [0, 0.5, 1]   (bin 0: [0, 0.5),  bin 1: [0.5, 1])
        y_bins = [0, 0.5, 1]   (bin 0: [0, 0.5),  bin 1: [0.5, 1])

    Points:
        (0.25, 0.25) → cell (0, 0)  →  probability = 0.5
        (0.75, 0.75) → cell (1, 1)  →  probability = 0.5
    """
    x_bins = torch.tensor([0.0, 0.5, 1.0])
    y_bins = torch.tensor([0.0, 0.5, 1.0])

    probs, _, _ = build_density_matrix(
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.25, 0.75]),
        x_bins=x_bins,
        y_bins=y_bins,
    )
    result = lookup_density_values(
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.25, 0.75]),
        probs,
        x_bins,
        y_bins,
    )

    assert torch.allclose(result, torch.tensor([0.5, 0.5]), atol=1e-6), (
        f"Expected [0.5, 0.5], got {result.tolist()}"
    )


def test_nan_input_returns_zero():
    """
    NaN query coordinates are invalid and must map to 0.0 in the output.
    This mirrors the `valid = torch.isfinite(x) & torch.isfinite(y)` guard.
    """
    x_bins = torch.linspace(0.0, 1.0, 6)
    y_bins = torch.linspace(0.0, 1.0, 6)
    probs, _, _ = build_density_matrix(
        torch.linspace(0.1, 0.9, 50),
        torch.linspace(0.1, 0.9, 50),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    result = lookup_density_values(
        torch.tensor([float("nan"), 0.5]),
        torch.tensor([0.5, 0.5]),
        probs,
        x_bins,
        y_bins,
    )

    assert result[0].item() == 0.0, f"Expected 0.0 for NaN, got {result[0].item()}"


def test_inf_input_returns_zero():
    """
    ±inf query coordinates are invalid and must map to 0.0.
    """
    x_bins = torch.linspace(0.0, 1.0, 6)
    y_bins = torch.linspace(0.0, 1.0, 6)
    probs, _, _ = build_density_matrix(
        torch.linspace(0.1, 0.9, 50),
        torch.linspace(0.1, 0.9, 50),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    result = lookup_density_values(
        torch.tensor([float("inf"), float("-inf")]),
        torch.tensor([0.5, 0.5]),
        probs,
        x_bins,
        y_bins,
    )

    assert result[0].item() == 0.0, f"Expected 0.0 for +inf, got {result[0].item()}"
    assert result[1].item() == 0.0, f"Expected 0.0 for -inf, got {result[1].item()}"


def test_all_nan_returns_all_zeros():
    """
    When every query point is NaN, the early-exit guard fires and the entire
    output tensor must be zero.
    """
    x_bins = torch.linspace(0.0, 1.0, 6)
    y_bins = torch.linspace(0.0, 1.0, 6)
    probs, _, _ = build_density_matrix(
        torch.linspace(0.1, 0.9, 50),
        torch.linspace(0.1, 0.9, 50),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    result = lookup_density_values(
        torch.full((5,), float("nan")),
        torch.full((5,), float("nan")),
        probs,
        x_bins,
        y_bins,
    )

    assert torch.all(result == 0.0), f"Expected all zeros, got {result.tolist()}"


def test_out_of_range_is_clamped_to_edge_bin():
    """
    A query point below x_min must be clamped to bin 0 — not zero-filled.

    Setup: all training data at (0.05, 0.45) → x-bin 0, y-bin 4.
    An in-range query at (0.05, 0.45) and an out-of-range query at
    (-999, 0.45) must both return the same non-zero probability after clamping.
    """
    x_bins = torch.linspace(0.0, 1.0, 11)
    y_bins = torch.linspace(0.0, 1.0, 11)
    probs, _, _ = build_density_matrix(
        torch.full((100,), 0.05),
        torch.full((100,), 0.45),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    in_range = lookup_density_values(torch.tensor([0.05]), torch.tensor([0.45]), probs, x_bins, y_bins)
    clamped = lookup_density_values(torch.tensor([-999.0]), torch.tensor([0.45]), probs, x_bins, y_bins)

    assert clamped.item() > 0.0, "Clamped out-of-range query returned 0 — expected non-zero"
    assert torch.isclose(in_range, clamped, atol=1e-6), (
        f"In-range ({in_range.item():.6f}) and clamped ({clamped.item():.6f}) differ"
    )


def test_output_length_matches_input_length():
    """
    The output tensor must have the same number of elements as the input,
    regardless of how many are valid, NaN, or out-of-range.
    """
    x_bins = torch.linspace(0.0, 10.0, 11)
    y_bins = torch.linspace(0.0, 10.0, 11)
    probs, _, _ = build_density_matrix(
        torch.linspace(1.0, 9.0, 100),
        torch.linspace(1.0, 9.0, 100),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    query_x = torch.tensor([1.0, 5.0, float("nan"), -999.0, 9.0])
    query_y = torch.tensor([1.0, 5.0, 5.0, 5.0, 9.0])
    result = lookup_density_values(query_x, query_y, probs, x_bins, y_bins)

    assert result.numel() == query_x.numel(), (
        f"Output length {result.numel()} != input length {query_x.numel()}"
    )


# ===========================================================================
# lookup_density_values — real-data test
# ===========================================================================


def test_returned_values_are_non_negative_on_real_data(real_tensors):
    """
    All probability values returned by lookup_density_values must be >= 0
    when queried with real gen-level feature tensors from Activation.sim.

    The gen tensors mix beta-decay and background events (in event order)
    and include NaN entries from events with zero hits.
    """
    t = real_tensors
    probs, x_bins, y_bins = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
    )
    result = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        probs,
        x_bins,
        y_bins,
    )

    assert torch.all(result >= 0.0), f"Found {(result < 0).sum().item()} negative probability values"


# ===========================================================================
# Physics tests — TRUE events (beta-decay) matrix structure
# ===========================================================================


def test_bdecay_angle_matrix_concentrated_at_low_deltaE(real_tensors, density_matrices, wandb_run):
    """
    The beta-decay (delta_E vs annihilation_angle) matrix must have most of
    its probability mass concentrated in the low-delta_E region.

    Physics basis: a true 511 keV annihilation photon deposits energy very
    close to 511 keV, so |sum(E) - 511| ≈ 0.  The 90th-percentile of the
    delta_E marginal must be reached with a smaller fraction of bins for
    bdecay than for background.

    W&B plot: bdecay heatmap (delta_E vs angle) with a vertical marker at the
    90th-percentile delta_E value.
    """
    dm = density_matrices

    bdecay_dE_marginal = dm["bdecay_angle"].sum(dim=1)  # P(delta_E) for bdecay
    bg_dE_marginal = dm["bg_angle"].sum(dim=1)  # P(delta_E) for bg

    bdecay_frac = ((bdecay_dE_marginal.cumsum(0) < 0.9).sum().item() + 1) / len(bdecay_dE_marginal)
    bg_frac = ((bg_dE_marginal.cumsum(0) < 0.9).sum().item() + 1) / len(bg_dE_marginal)

    assert bdecay_frac < bg_frac, (
        f"Expected bdecay to reach 90% mass in fewer bins than bg "
        f"(bdecay={bdecay_frac:.3f}, bg={bg_frac:.3f})"
    )

    # --- W&B plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_heatmap(
        ax,
        dm["bdecay_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
        "Beta-decay: delta_E vs Annihilation Angle",
        _LABEL_DE,
        _LABEL_ANGLE,
    )
    p90_dE = float(_to_np(dm["x_bins_angle"])[int(bdecay_frac * len(bdecay_dE_marginal))])
    ax.axvline(x=max(p90_dE, 1e-3), color="red", ls="--", label=f"90th pct delta_E = {p90_dE:.2f} keV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/bdecay_angle_deltaE_concentration": wandb.Image(fig)})
    plt.close(fig)


def test_bdecay_arm_matrix_peaked_near_zero(real_tensors, density_matrices, wandb_run):
    """
    The beta-decay (delta_E vs ARM) matrix must have more than half of its
    ARM probability mass in the lower half of ARM bins (ARM near zero).

    Physics basis: a correctly reconstructed Compton event has a small
    angular residual (ARM ≈ 0).  Beta-decay annihilation events, which
    produce 511 keV photons, should exhibit good Compton consistency.

    W&B plot: bdecay ARM heatmap with a horizontal marker at the ARM median.
    """
    dm = density_matrices

    bdecay_arm_marginal = dm["bdecay_arm"].sum(dim=0)  # P(ARM) for bdecay
    n_bins = len(bdecay_arm_marginal)
    cumsum = bdecay_arm_marginal.cumsum(0)

    # Fraction of ARM bins needed to capture 50% of bdecay mass
    median_bin = (cumsum < 0.5).sum().item()
    frac_to_median = median_bin / n_bins

    assert frac_to_median < 0.5, (
        f"Expected bdecay ARM median in lower half of bins, but it required {frac_to_median:.1%} of all bins"
    )

    # --- W&B plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_heatmap(
        ax,
        dm["bdecay_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        "Beta-decay: delta_E vs ARM",
        _LABEL_DE,
        _LABEL_ARM,
        cmap="plasma",
    )
    median_arm = float(_to_np(dm["y_bins_arm"])[median_bin])
    ax.axhline(y=median_arm, color="red", ls="--", label=f"ARM median = {median_arm:.3f} rad")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/bdecay_arm_peaked_near_zero": wandb.Image(fig)})
    plt.close(fig)


# ===========================================================================
# Physics tests — FALSE events (background) matrix structure
# ===========================================================================


def test_bg_angle_matrix_more_spread_than_signal(real_tensors, density_matrices, wandb_run):
    """
    The background delta_E marginal (from the angle-feature matrix) must
    require more bins than the beta-decay marginal to capture 90% of mass,
    confirming that background energy depositions are more spread out.

    W&B plot: overlaid delta_E CDFs for signal and background.
    """
    dm = density_matrices

    bdecay_marginal = _to_np(dm["bdecay_angle"].sum(dim=1))
    bg_marginal = _to_np(dm["bg_angle"].sum(dim=1))
    centers = _bin_centers(dm["x_bins_angle"])

    bdecay_frac = (np.cumsum(bdecay_marginal) < 0.9).sum() / len(bdecay_marginal)
    bg_frac = (np.cumsum(bg_marginal) < 0.9).sum() / len(bg_marginal)

    assert bg_frac > bdecay_frac, (
        f"Expected background to need more bins for 90% CDF than signal "
        f"(bg={bg_frac:.3f}, bdecay={bdecay_frac:.3f})"
    )

    # --- W&B plot: overlaid CDFs ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    _plot_1d_overlay(
        axes[0],
        np.cumsum(bdecay_marginal),
        np.cumsum(bg_marginal),
        centers,
        _LABEL_DE,
        "delta_E cumulative distribution (angle pair)",
    )
    axes[0].axhline(0.9, color="gray", ls=":", label="90%")
    axes[0].legend()

    _plot_1d_overlay(
        axes[1],
        bdecay_marginal,
        bg_marginal,
        centers,
        _LABEL_DE,
        "delta_E marginal PDF (angle pair)",
    )

    fig.tight_layout()
    wandb.log({"test/bg_angle_deltaE_spread": wandb.Image(fig)})
    plt.close(fig)


def test_bg_arm_matrix_not_peaked_at_zero(real_tensors, density_matrices, wandb_run):
    """
    The background ARM marginal must have less mass in the lowest quarter of
    ARM bins than the beta-decay ARM marginal does.

    Physics basis: background Compton sequences are less consistent with 511
    keV kinematics, so their ARM values are shifted toward larger values
    compared to true annihilation events.

    W&B plot: overlaid ARM PDFs (signal vs background) + heatmaps.
    """
    dm = density_matrices

    bdecay_arm_marginal = _to_np(dm["bdecay_arm"].sum(dim=0))
    bg_arm_marginal = _to_np(dm["bg_arm"].sum(dim=0))
    n_bins = len(bdecay_arm_marginal)
    quarter = n_bins // 4

    bdecay_low_arm_mass = bdecay_arm_marginal[:quarter].sum()
    bg_low_arm_mass = bg_arm_marginal[:quarter].sum()

    assert bdecay_low_arm_mass > bg_low_arm_mass, (
        f"Expected bdecay to have more mass near ARM=0 than background "
        f"(bdecay={bdecay_low_arm_mass:.4f}, bg={bg_low_arm_mass:.4f})"
    )

    # --- W&B plot: side-by-side heatmaps + marginal overlay ---
    centers = _bin_centers(dm["y_bins_arm"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_heatmap(
        axes[0],
        dm["bdecay_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        "Beta-decay: delta_E vs ARM",
        _LABEL_DE,
        _LABEL_ARM,
        cmap="viridis",
    )
    _plot_heatmap(
        axes[1],
        dm["bg_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        "Background: delta_E vs ARM",
        _LABEL_DE,
        _LABEL_ARM,
        cmap="plasma",
    )
    _plot_1d_overlay(axes[2], bdecay_arm_marginal, bg_arm_marginal, centers, _LABEL_ARM, "ARM marginal PDF")

    fig.tight_layout()
    wandb.log({"test/bg_arm_not_peaked_at_zero": wandb.Image(fig)})
    plt.close(fig)


# ===========================================================================
# Physics tests — Entropy: signal is always less entropic than background
# ===========================================================================


def test_signal_entropy_lower_than_background_angle_feature(real_tensors, density_matrices, wandb_run):
    """
    The Shannon entropy of the beta-decay matrix must be strictly lower than
    that of the background matrix for the (delta_E, annihilation_angle) pair.

    Lower entropy = more concentrated probability mass = more discriminative.
    This is a necessary condition for the Bayesian classifier to work: a
    uniform signal matrix would carry no information over the background.

    W&B plot: side-by-side heatmaps with entropy values in the titles.
    """
    dm = density_matrices

    h_bdecay = _entropy(dm["bdecay_angle"])
    h_bg = _entropy(dm["bg_angle"])

    assert h_bdecay < h_bg, (
        f"Expected H(bdecay) < H(bg) for angle feature pair (H_bdecay={h_bdecay:.3f}, H_bg={h_bg:.3f} nats)"
    )

    # --- W&B plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_heatmap(
        axes[0],
        dm["bdecay_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
        f"Beta-decay  H={h_bdecay:.2f} nats",
        _LABEL_DE,
        _LABEL_ANGLE,
    )
    _plot_heatmap(
        axes[1],
        dm["bg_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
        f"Background  H={h_bg:.2f} nats",
        _LABEL_DE,
        _LABEL_ANGLE,
        cmap="plasma",
    )
    fig.suptitle("Entropy comparison — angle feature pair", fontsize=12)
    fig.tight_layout()
    wandb.log({"test/entropy_angle_signal_vs_bg": wandb.Image(fig)})
    plt.close(fig)


def test_signal_entropy_lower_than_background_arm_feature(real_tensors, density_matrices, wandb_run):
    """
    The Shannon entropy of the beta-decay matrix must be strictly lower than
    that of the background matrix for the (delta_E, ARM) pair.

    W&B plot: side-by-side ARM heatmaps with entropy values in the titles.
    """
    dm = density_matrices

    h_bdecay = _entropy(dm["bdecay_arm"])
    h_bg = _entropy(dm["bg_arm"])

    assert h_bdecay < h_bg, (
        f"Expected H(bdecay) < H(bg) for ARM feature pair (H_bdecay={h_bdecay:.3f}, H_bg={h_bg:.3f} nats)"
    )

    # --- W&B plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_heatmap(
        axes[0],
        dm["bdecay_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        f"Beta-decay  H={h_bdecay:.2f} nats",
        _LABEL_DE,
        _LABEL_ARM,
    )
    _plot_heatmap(
        axes[1],
        dm["bg_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        f"Background  H={h_bg:.2f} nats",
        _LABEL_DE,
        _LABEL_ARM,
        cmap="plasma",
    )
    fig.suptitle("Entropy comparison — ARM feature pair", fontsize=12)
    fig.tight_layout()
    wandb.log({"test/entropy_arm_signal_vs_bg": wandb.Image(fig)})
    plt.close(fig)


# ===========================================================================
# Physics tests — Distinguishability: TV distance + log-ratio sign
# ===========================================================================


def test_angle_matrices_distinguishable_by_total_variation(real_tensors, density_matrices, wandb_run):
    """
    The total variation distance between the signal and background
    (delta_E, annihilation_angle) matrices must exceed 0.1.

    TV = 0 means identical distributions; TV = 1 means no shared support.
    A threshold of 0.1 is conservative: any meaningful physical separation
    between the two classes will comfortably exceed it.

    W&B plot: |P_bdecay − P_bg| difference heatmap.
    """
    dm = density_matrices

    tv = _total_variation(dm["bdecay_angle"], dm["bg_angle"])

    assert tv > 0.1, f"Expected TV distance > 0.1 for angle matrices, got {tv:.4f}"

    # --- W&B plot ---
    diff = torch.abs(dm["bdecay_angle"] - dm["bg_angle"])
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_heatmap(
        ax,
        diff,
        dm["x_bins_angle"],
        dm["y_bins_angle"],
        f"|P_bdecay − P_bg| (angle pair)   TV = {tv:.3f}",
        _LABEL_DE,
        _LABEL_ANGLE,
        cmap="hot",
    )
    fig.tight_layout()
    wandb.log({"test/tv_distance_angle": wandb.Image(fig)})
    plt.close(fig)


def test_arm_matrices_distinguishable_by_total_variation(real_tensors, density_matrices, wandb_run):
    """
    The total variation distance between the signal and background
    (delta_E, ARM) matrices must exceed 0.1.

    W&B plot: |P_bdecay − P_bg| difference heatmap for the ARM pair.
    """
    dm = density_matrices

    tv = _total_variation(dm["bdecay_arm"], dm["bg_arm"])

    assert tv > 0.1, f"Expected TV distance > 0.1 for ARM matrices, got {tv:.4f}"

    # --- W&B plot ---
    diff = torch.abs(dm["bdecay_arm"] - dm["bg_arm"])
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_heatmap(
        ax,
        diff,
        dm["x_bins_arm"],
        dm["y_bins_arm"],
        f"|P_bdecay − P_bg| (ARM pair)   TV = {tv:.3f}",
        _LABEL_DE,
        _LABEL_ARM,
        cmap="hot",
    )
    fig.tight_layout()
    wandb.log({"test/tv_distance_arm": wandb.Image(fig)})
    plt.close(fig)


def test_log_ratio_positive_in_signal_region_angle(real_tensors, density_matrices, wandb_run):
    """
    In the signal-enriched region of the (delta_E, annihilation_angle) space,
    the log-likelihood ratio log(P_signal / P_bg) must be positive for the
    majority of cells that carry significant beta-decay probability mass.

    "Significant" is defined as cells where bdecay probability > 1% of the
    maximum bdecay probability.  This threshold filters out near-empty cells
    where the log ratio is noise-dominated.

    W&B plot: log-ratio heatmap (diverging red/blue colormap).
    """
    dm = density_matrices
    eps = 1e-10
    bdecay = dm["bdecay_angle"]
    bg = dm["bg_angle"]

    threshold = 0.01 * bdecay.max()
    signal_mask = bdecay > threshold
    log_ratio = torch.log(bdecay + eps) - torch.log(bg + eps)

    frac_positive = (log_ratio[signal_mask] > 0).float().mean().item()

    assert frac_positive > 0.5, (
        f"Expected majority of signal-region cells to have positive log-ratio "
        f"(angle pair), got {frac_positive:.1%}"
    )

    # --- W&B plot: side-by-side signal vs background ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_heatmap(axes[0], bdecay, dm["x_bins_angle"], dm["y_bins_angle"],
                  "Beta-decay density (angle pair)", _LABEL_DE, _LABEL_ANGLE)
    _plot_heatmap(axes[1], bg, dm["x_bins_angle"], dm["y_bins_angle"],
                  "Background density (angle pair)", _LABEL_DE, _LABEL_ANGLE, cmap="plasma")
    fig.suptitle(f"Signal-region cells with P_signal > P_bg: {frac_positive:.1%}", fontsize=11)
    fig.tight_layout()
    wandb.log({"test/signal_vs_bg_angle": wandb.Image(fig)})
    plt.close(fig)


def test_log_ratio_positive_in_signal_region_arm(real_tensors, density_matrices, wandb_run):
    """
    In the signal-enriched region of the (delta_E, ARM) space, the
    log-likelihood ratio must be positive for the majority of cells carrying
    significant beta-decay probability mass.

    W&B plot: log-ratio heatmap for the ARM feature pair.
    """
    dm = density_matrices
    eps = 1e-10
    bdecay = dm["bdecay_arm"]
    bg = dm["bg_arm"]

    threshold = 0.01 * bdecay.max()
    signal_mask = bdecay > threshold
    log_ratio = torch.log(bdecay + eps) - torch.log(bg + eps)

    frac_positive = (log_ratio[signal_mask] > 0).float().mean().item()

    assert frac_positive > 0.5, (
        f"Expected majority of signal-region cells to have positive log-ratio "
        f"(ARM pair), got {frac_positive:.1%}"
    )

    # --- W&B plot: side-by-side signal vs background ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_heatmap(axes[0], bdecay, dm["x_bins_arm"], dm["y_bins_arm"],
                  "Beta-decay density (ARM pair)", _LABEL_DE, _LABEL_ARM)
    _plot_heatmap(axes[1], bg, dm["x_bins_arm"], dm["y_bins_arm"],
                  "Background density (ARM pair)", _LABEL_DE, _LABEL_ARM, cmap="plasma")
    fig.suptitle(f"Signal-region cells with P_signal > P_bg: {frac_positive:.1%}", fontsize=11)
    fig.tight_layout()
    wandb.log({"test/signal_vs_bg_arm": wandb.Image(fig)})
    plt.close(fig)


# ===========================================================================
# Physics tests — Ground-truth accuracy: event-level density lookup
# ===========================================================================


def test_majority_of_true_events_higher_bdecay_density_angle(real_tensors, density_matrices, wandb_run):
    """
    For events the simulation labels as beta-decay (ground_truths == True),
    the majority must fall in cells where P_bdecay > P_bg in the
    (delta_E, annihilation_angle) matrix.

    Events with NaN features (zero-hit events) get 0 from both lookups
    and are excluded from the accuracy calculation.

    W&B plot: true-event scatter overlaid on the log-ratio heatmap.
    """
    t, dm = real_tensors, density_matrices
    gt = t["ground_truths"]

    n_beta = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        dm["bdecay_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
    )
    n_bg = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        dm["bg_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
    )

    valid = gt & ((n_beta > 0) | (n_bg > 0))
    accuracy = (n_beta[valid] > n_bg[valid]).float().mean().item()
    n_valid = valid.sum().item()

    assert accuracy > 0.55, (
        f"Expected >55% of true bdecay events to have higher bdecay density "
        f"(angle pair), got {accuracy:.1%} (n={n_valid})"
    )

    # --- W&B plot: true events scattered on the bdecay density heatmap ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_heatmap(ax, dm["bdecay_angle"], dm["x_bins_angle"], dm["y_bins_angle"],
                  f"True bdecay events on signal density (angle)  acc={accuracy:.1%}",
                  _LABEL_DE, _LABEL_ANGLE)
    dE_true    = _to_np(t["gen_tensor_delta_E"][gt])[:500]
    angle_true = _to_np(t["gen_tensor_annihilation_angle"][gt])[:500]
    ax.scatter(dE_true, angle_true, s=4, c="red", alpha=0.5, label="True events (n≤500)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/true_events_angle": wandb.Image(fig)})
    plt.close(fig)


def test_majority_of_true_events_higher_bdecay_density_arm(real_tensors, density_matrices, wandb_run):
    """
    For events the simulation labels as beta-decay, the majority must fall in
    cells where P_bdecay > P_bg in the (delta_E, ARM) matrix.

    W&B plot: true-event scatter overlaid on the ARM log-ratio heatmap.
    """
    t, dm = real_tensors, density_matrices
    gt = t["ground_truths"]

    n_beta = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        dm["bdecay_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
    )
    n_bg = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        dm["bg_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
    )

    valid = gt & ((n_beta > 0) | (n_bg > 0))
    accuracy = (n_beta[valid] > n_bg[valid]).float().mean().item()
    n_valid = valid.sum().item()

    assert accuracy > 0.55, (
        f"Expected >55% of true bdecay events to have higher bdecay density "
        f"(ARM pair), got {accuracy:.1%} (n={n_valid})"
    )

    # --- W&B plot: true events scattered on the bdecay density heatmap ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_heatmap(ax, dm["bdecay_arm"], dm["x_bins_arm"], dm["y_bins_arm"],
                  f"True bdecay events on signal density (ARM)  acc={accuracy:.1%}",
                  _LABEL_DE, _LABEL_ARM)
    dE_true  = _to_np(t["gen_tensor_delta_E"][gt])[:500]
    arm_true = _to_np(t["gen_tensor_arm"][gt])[:500]
    ax.scatter(dE_true, arm_true, s=4, c="red", alpha=0.5, label="True events (n≤500)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/true_events_arm": wandb.Image(fig)})
    plt.close(fig)


def test_majority_of_false_events_higher_bg_density_angle(real_tensors, density_matrices, wandb_run):
    """
    For events the simulation labels as background (ground_truths == False),
    the majority must fall in cells where P_bg > P_bdecay in the
    (delta_E, annihilation_angle) matrix.

    W&B plot: background-event scatter overlaid on the angle log-ratio heatmap.
    """
    t, dm = real_tensors, density_matrices
    gt = t["ground_truths"]

    n_beta = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        dm["bdecay_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
    )
    n_bg = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        dm["bg_angle"],
        dm["x_bins_angle"],
        dm["y_bins_angle"],
    )

    valid = ~gt & ((n_beta > 0) | (n_bg > 0))
    accuracy = (n_bg[valid] > n_beta[valid]).float().mean().item()
    n_valid = valid.sum().item()

    assert accuracy > 0.55, (
        f"Expected >55% of background events to have higher bg density "
        f"(angle pair), got {accuracy:.1%} (n={n_valid})"
    )

    # --- W&B plot: background events scattered on the bg density heatmap ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_heatmap(ax, dm["bg_angle"], dm["x_bins_angle"], dm["y_bins_angle"],
                  f"Background events on bg density (angle)  acc={accuracy:.1%}",
                  _LABEL_DE, _LABEL_ANGLE, cmap="plasma")
    dE_bg    = _to_np(t["gen_tensor_delta_E"][~gt])[:500]
    angle_bg = _to_np(t["gen_tensor_annihilation_angle"][~gt])[:500]
    ax.scatter(dE_bg, angle_bg, s=2, c="white", alpha=0.3, label="Background events (n≤500)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/false_events_angle": wandb.Image(fig)})
    plt.close(fig)


def test_majority_of_false_events_higher_bg_density_arm(real_tensors, density_matrices, wandb_run):
    """
    For events the simulation labels as background, the majority must fall in
    cells where P_bg > P_bdecay in the (delta_E, ARM) matrix.

    W&B plot: background-event scatter overlaid on the ARM log-ratio heatmap.
    """
    t, dm = real_tensors, density_matrices
    gt = t["ground_truths"]

    n_beta = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        dm["bdecay_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
    )
    n_bg = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        dm["bg_arm"],
        dm["x_bins_arm"],
        dm["y_bins_arm"],
    )

    valid = ~gt & ((n_beta > 0) | (n_bg > 0))
    accuracy = (n_bg[valid] > n_beta[valid]).float().mean().item()
    n_valid = valid.sum().item()

    assert accuracy > 0.55, (
        f"Expected >55% of background events to have higher bg density "
        f"(ARM pair), got {accuracy:.1%} (n={n_valid})"
    )

    # --- W&B plot: background events scattered on the bg density heatmap ---
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_heatmap(ax, dm["bg_arm"], dm["x_bins_arm"], dm["y_bins_arm"],
                  f"Background events on bg density (ARM)  acc={accuracy:.1%}",
                  _LABEL_DE, _LABEL_ARM, cmap="plasma")
    dE_bg  = _to_np(t["gen_tensor_delta_E"][~gt])[:500]
    arm_bg = _to_np(t["gen_tensor_arm"][~gt])[:500]
    ax.scatter(dE_bg, arm_bg, s=2, c="white", alpha=0.3, label="Background events (n≤500)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    wandb.log({"test/false_events_arm": wandb.Image(fig)})
    plt.close(fig)
