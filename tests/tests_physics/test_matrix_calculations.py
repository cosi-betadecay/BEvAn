"""
Unit tests for physics/matrix_calculations.py.

Covers:
    - build_density_matrix  (10 tests)
    - lookup_density_values (7 tests)

All tests are self-contained: no MEGAlib / ROOT / W&B dependencies required.
"""

import pytest
import torch

from physics.matrix_calculations import build_density_matrix, lookup_density_values

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(x_vals, y_vals, x_bins, y_bins):
    """Convenience wrapper that builds a density matrix with fixed bin edges."""
    x = torch.tensor(x_vals, dtype=torch.float32)
    y = torch.tensor(y_vals, dtype=torch.float32)
    return build_density_matrix(x, y, x_bins=x_bins, y_bins=y_bins)


# ===========================================================================
# build_density_matrix
# ===========================================================================


def test_probabilities_sum_to_one():
    """
    The normalized probability matrix must always sum to exactly 1.0.

    build_density_matrix divides the raw counts by their total, so any
    finite, non-empty input should produce probabilities that sum to 1.
    """
    torch.manual_seed(0)
    x = torch.randn(500)
    y = torch.randn(500)

    probs, _, _ = build_density_matrix(x, y, n_bins_x=50, n_bins_y=50)

    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_output_shape():
    """
    The probability matrix shape must equal (n_bins_x, n_bins_y).
    Returned x_bins must have n_bins_x + 1 elements (bin edges),
    and y_bins must have n_bins_y + 1 elements.
    """
    x = torch.linspace(0.0, 10.0, 200)
    y = torch.linspace(0.0, 10.0, 200)

    probs, x_bins, y_bins = build_density_matrix(x, y, n_bins_x=20, n_bins_y=30)

    assert probs.shape == (20, 30), f"Expected (20, 30), got {tuple(probs.shape)}"
    assert x_bins.numel() == 21, f"Expected 21 x-bin edges, got {x_bins.numel()}"
    assert y_bins.numel() == 31, f"Expected 31 y-bin edges, got {y_bins.numel()}"


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

    probs, _, _ = build_density_matrix(x, y, n_bins_x=100, n_bins_y=100)

    assert probs.max().item() > 0.9, f"Expected max cell probability > 0.9, got {probs.max().item():.4f}"


def test_shared_bins_produce_consistent_shapes():
    """
    A common usage pattern: build bins from a combined (signal + background)
    dataset and re-use those bins for the individual signal and background
    matrices.

    Both downstream matrices must have the same shape as the combined
    matrix, proving that shared bin edges produce a consistent coordinate
    system.
    """
    torch.manual_seed(2)
    combined_x = torch.cat([torch.randn(100) + 0.0, torch.randn(100) + 5.0])
    combined_y = torch.cat([torch.randn(100) + 0.0, torch.randn(100) + 5.0])

    _, x_bins, y_bins = build_density_matrix(combined_x, combined_y, n_bins_x=25, n_bins_y=25)
    expected_shape = (25, 25)

    signal_x = torch.randn(80)
    signal_y = torch.randn(80)
    signal_probs, _, _ = build_density_matrix(signal_x, signal_y, x_bins=x_bins, y_bins=y_bins)

    bg_x = torch.randn(80) + 5.0
    bg_y = torch.randn(80) + 5.0
    bg_probs, _, _ = build_density_matrix(bg_x, bg_y, x_bins=x_bins, y_bins=y_bins)

    assert signal_probs.shape == expected_shape, (
        f"Signal matrix shape {signal_probs.shape} != expected {expected_shape}"
    )
    assert bg_probs.shape == expected_shape, (
        f"Background matrix shape {bg_probs.shape} != expected {expected_shape}"
    )


# ===========================================================================
# lookup_density_values
# ===========================================================================


def test_known_point_returns_correct_probability():
    """
    Round-trip accuracy: build a 2×2 probability matrix from two points that
    fall into distinct, predictable cells, then verify that a lookup at
    each point returns the exact stored probability.

    Matrix layout:
        x_bins = [0, 0.5, 1]   (bin 0: [0, 0.5),  bin 1: [0.5, 1])
        y_bins = [0, 0.5, 1]   (bin 0: [0, 0.5),  bin 1: [0.5, 1])

    Points:
        (0.25, 0.25) → cell (0, 0)  →  probability = 0.5
        (0.75, 0.75) → cell (1, 1)  →  probability = 0.5
    """
    x_bins = torch.tensor([0.0, 0.5, 1.0])
    y_bins = torch.tensor([0.0, 0.5, 1.0])

    x = torch.tensor([0.25, 0.75])
    y = torch.tensor([0.25, 0.75])
    probs, _, _ = build_density_matrix(x, y, x_bins=x_bins, y_bins=y_bins)

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
    NaN query coordinates are considered invalid and must map to a
    probability of 0.0 in the output.

    This mirrors the filtering logic inside lookup_density_values:
    `valid = torch.isfinite(x) & torch.isfinite(y)`.
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

    assert result[0].item() == 0.0, f"Expected 0.0 for NaN input, got {result[0].item()}"


def test_inf_input_returns_zero():
    """
    ±inf query coordinates are invalid and must map to 0.0.

    Infinity is not finite, so `torch.isfinite` returns False for it, and
    the corresponding output slot is left at the initial fill value of 0.0.
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
    When every query point is NaN, the early-exit guard
    (`if not torch.any(valid): return values`) must fire and the entire
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

    nan_x = torch.full((5,), float("nan"))
    nan_y = torch.full((5,), float("nan"))
    result = lookup_density_values(nan_x, nan_y, probs, x_bins, y_bins)

    assert torch.all(result == 0.0), f"Expected all zeros, got {result.tolist()}"


def test_out_of_range_is_clamped_to_edge_bin():
    """
    A query point whose x coordinate is far below x_min must be clamped to
    the first bin (index 0) rather than returning 0 or raising an error.

    Setup: all data sits in x-bin 0 (x ≈ 0.05), y-bin 4 (y ≈ 0.5).
    An in-range query at (0.05, 0.5) and an out-of-range query at
    (-999.0, 0.5) must return the same non-zero probability because both
    are resolved to x_idx = 0 after clamping.
    """
    x_bins = torch.linspace(0.0, 1.0, 11)  # 10 bins, width 0.1 each
    y_bins = torch.linspace(0.0, 1.0, 11)

    x = torch.full((100,), 0.05)  # all points land in x-bin 0
    y = torch.full((100,), 0.45)  # all points land in y-bin 4
    probs, _, _ = build_density_matrix(x, y, x_bins=x_bins, y_bins=y_bins)

    in_range = lookup_density_values(torch.tensor([0.05]), torch.tensor([0.45]), probs, x_bins, y_bins)
    clamped = lookup_density_values(torch.tensor([-999.0]), torch.tensor([0.45]), probs, x_bins, y_bins)

    assert clamped.item() > 0.0, "Clamped out-of-range query returned 0 — expected non-zero"
    assert torch.isclose(in_range, clamped, atol=1e-6), (
        f"In-range ({in_range.item():.6f}) and clamped ({clamped.item():.6f}) "
        "results differ — clamping is not working correctly"
    )


def test_output_length_matches_input_length():
    """
    The output tensor must have the same number of elements as the input x
    (and y) tensor, regardless of how many points are valid or out-of-range.

    This ensures callers can safely index the output with the same indices
    they used for the input.
    """
    x_bins = torch.linspace(0.0, 10.0, 11)
    y_bins = torch.linspace(0.0, 10.0, 11)
    probs, _, _ = build_density_matrix(
        torch.linspace(1.0, 9.0, 100),
        torch.linspace(1.0, 9.0, 100),
        x_bins=x_bins,
        y_bins=y_bins,
    )

    # Mix of valid, NaN, and out-of-range points
    query_x = torch.tensor([1.0, 5.0, float("nan"), -999.0, 9.0])
    query_y = torch.tensor([1.0, 5.0, 5.0, 5.0, 9.0])
    result = lookup_density_values(query_x, query_y, probs, x_bins, y_bins)

    assert result.numel() == query_x.numel(), (
        f"Output length {result.numel()} != input length {query_x.numel()}"
    )


def test_returned_values_are_non_negative():
    """
    All returned probability values must be >= 0.

    Since the probability matrix contains normalized counts, every stored
    value is in [0, 1].  Invalid inputs (NaN/inf) map to 0.0.  Neither
    clamped out-of-range inputs nor valid in-range inputs should ever
    produce a negative value.
    """
    torch.manual_seed(3)
    x_bins = torch.linspace(0.0, 10.0, 51)
    y_bins = torch.linspace(0.0, 10.0, 51)
    probs, _, _ = build_density_matrix(
        torch.rand(300) * 10.0,
        torch.rand(300) * 10.0,
        x_bins=x_bins,
        y_bins=y_bins,
    )

    query_x = torch.cat([torch.rand(50) * 10.0, torch.tensor([float("nan"), -5.0, 15.0])])
    query_y = torch.cat([torch.rand(50) * 10.0, torch.tensor([5.0, 5.0, 5.0])])
    result = lookup_density_values(query_x, query_y, probs, x_bins, y_bins)

    assert torch.all(result >= 0.0), f"Found negative probability values: {result[result < 0].tolist()}"
