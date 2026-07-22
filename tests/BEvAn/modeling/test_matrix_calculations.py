import pytest
import torch

from modeling.matrix_calculations import (
    build_default_edges,
    build_density_matrix,
    build_density_matrix_1d,
    lookup_density_values,
    lookup_density_values_1d,
)


def test_build_default_edges_linear():
    edges = build_default_edges(torch.tensor([0.0, 10.0]), n_bins=5, spacing="linear", log_floor=None)
    assert torch.allclose(edges, torch.linspace(0.0, 10.0, 6))


def test_build_default_edges_degenerate_range():
    edges = build_default_edges(torch.tensor([3.0, 3.0]), n_bins=4, spacing="linear", log_floor=None)
    assert edges.numel() == 5
    assert edges[-1] > edges[0]


def test_build_default_edges_log_floor():
    edges = build_default_edges(torch.tensor([1e-6, 1.0]), n_bins=3, spacing="log", log_floor=1e-3)
    assert edges[0].item() == pytest.approx(1e-3)
    assert edges[-1].item() == pytest.approx(1.0)


def test_build_default_edges_rejects_unknown_spacing_and_no_positives():
    with pytest.raises(ValueError, match="spacing"):
        build_default_edges(torch.tensor([1.0]), 2, "cubic", None)
    with pytest.raises(ValueError, match="positive"):
        build_default_edges(torch.tensor([-1.0, 0.0]), 2, "log", None)


def test_density_1d_normalized_and_lookup_roundtrip():
    x = torch.tensor([0.5, 1.5, 1.5, 2.5])
    probs, edges = build_density_matrix_1d(x, x_bins=torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert probs.sum().item() == pytest.approx(1.0)
    assert torch.allclose(probs, torch.tensor([0.25, 0.5, 0.25]))
    values = lookup_density_values_1d(torch.tensor([0.5, 1.5, 2.5]), probs, edges)
    assert torch.allclose(values, torch.tensor([0.25, 0.5, 0.25]))


def test_density_1d_clamps_out_of_range_and_zeroes_nan():
    probs, edges = build_density_matrix_1d(torch.tensor([0.5, 1.5]), x_bins=torch.tensor([0.0, 1.0, 2.0]))
    values = lookup_density_values_1d(torch.tensor([-5.0, 10.0, float("nan")]), probs, edges)
    assert values[0].item() == pytest.approx(probs[0].item())
    assert values[1].item() == pytest.approx(probs[-1].item())
    assert values[2].item() == 0.0


def test_density_1d_smoothing_adds_pseudocounts():
    probs, _ = build_density_matrix_1d(
        torch.tensor([0.5]), x_bins=torch.tensor([0.0, 1.0, 2.0]), smoothing=1.0
    )
    assert torch.allclose(probs, torch.tensor([2.0 / 3.0, 1.0 / 3.0]))


def test_density_2d_build_and_lookup():
    x = torch.tensor([0.5, 0.5, 1.5])
    y = torch.tensor([0.5, 1.5, 1.5])
    edges = torch.tensor([0.0, 1.0, 2.0])
    probs, xe, ye = build_density_matrix(x, y, x_bins=edges, y_bins=edges)
    assert probs.sum().item() == pytest.approx(1.0)
    assert probs[0, 0].item() == pytest.approx(1.0 / 3.0)
    values = lookup_density_values(x, y, probs, xe, ye)
    assert torch.allclose(values, torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))


def test_density_2d_drops_nonfinite_rows_on_build():
    x = torch.tensor([0.5, float("nan")])
    y = torch.tensor([0.5, 0.5])
    edges = torch.tensor([0.0, 1.0])
    probs, _, _ = build_density_matrix(x, y, x_bins=edges, y_bins=edges)
    assert probs.item() == pytest.approx(1.0)


def test_density_2d_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        build_density_matrix(torch.tensor([]), torch.tensor([]))
