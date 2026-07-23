from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from utils.plots import (
    axis_meta,
    density_term_features,
    plot_confusion_matrix,
    plot_density_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_sky_backprojection,
    pr_curve,
    roc_curve,
    to_numpy,
)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def term_1d() -> dict:
    """Return a sample 1-D density term for plotting tests."""
    return {
        "kind": "1d",
        "xfeat": "delta_E",
        "x_bins": torch.tensor([0.1, 1.0, 10.0]),
        "beta": torch.tensor([0.75, 0.25]),
        "bg": torch.tensor([0.25, 0.75]),
    }


def term_2d() -> dict:
    """Return a sample 2-D density term for plotting tests."""
    return {
        "kind": "2d",
        "xfeat": "delta_E",
        "yfeat": "arm",
        "x_bins": torch.tensor([0.1, 1.0, 10.0]),
        "y_bins": torch.tensor([-1.0, 0.0, 1.0]),
        "beta": torch.tensor([[0.5, 0.25], [0.25, 0.0]]),
        "bg": torch.tensor([[0.0, 0.25], [0.25, 0.5]]),
    }


def test_to_numpy_handles_tensors_and_lists() -> None:
    """Convert both tensors and nested lists to float numpy arrays."""
    assert to_numpy(torch.tensor([1.0, 2.0])).tolist() == [1.0, 2.0]
    array = to_numpy([[1, 2], [3, 4]])
    assert array.dtype == float
    assert array.shape == (2, 2)


def test_axis_meta_known_and_fallback() -> None:
    """Return known axis metadata and a linear fallback for unknown features."""
    assert axis_meta("delta_E")["scale"] == "log"
    assert axis_meta("mystery") == {"label": "mystery", "name": "mystery", "scale": "linear"}


def test_density_term_features_slugs() -> None:
    """Build feature slugs from 1-D and 2-D density terms."""
    assert density_term_features(term_1d()) == "delta_E"
    assert density_term_features(term_2d()) == "delta_E_arm"


def test_roc_curve_hand_values() -> None:
    """Match roc_curve output against hand-computed values."""
    fpr, tpr, auc = roc_curve([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0])
    assert auc == pytest.approx(1.0)
    assert fpr.tolist() == [0.0, 0.0, 0.0, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]


def test_roc_curve_ties_and_degenerate() -> None:
    """Handle tied scores and a single-class degenerate case (NaN AUC)."""
    fpr, tpr, auc = roc_curve([1.0, 1.0, 0.0, 0.0], [1, 0, 1, 0])
    assert auc == pytest.approx(0.5)
    assert fpr.tolist() == [0.0, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 1.0]
    assert np.isnan(roc_curve([1.0, 2.0], [1, 1])[2])


def test_pr_curve_hand_values() -> None:
    """Match pr_curve output against hand-computed values, NaN when no positives."""
    recall, precision, ap = pr_curve([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0])
    assert ap == pytest.approx(1.0)
    assert recall.tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]
    assert precision.tolist() == [1.0, 1.0, 1.0, 2 / 3, 0.5]
    assert np.isnan(pr_curve([1.0, 2.0], [0, 0])[2])


def test_plot_confusion_matrix_returns_figure() -> None:
    """Return a matplotlib Figure from plot_confusion_matrix."""
    assert isinstance(plot_confusion_matrix(1, 2, 3, 4), Figure)


def test_plot_score_curves_return_figures() -> None:
    """Return Figures from the ROC and PR plotting helpers."""
    scores, labels = torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0])
    assert isinstance(plot_roc_curve(scores, labels), Figure)
    assert isinstance(plot_pr_curve(scores, labels), Figure)


def test_plot_density_matrix_dispatches_on_kind() -> None:
    """Dispatch density plotting on term kind and reject unknown kinds."""
    assert isinstance(plot_density_matrix(term_1d()), Figure)
    assert isinstance(plot_density_matrix(term_2d()), Figure)
    with pytest.raises(ValueError, match="kind"):
        plot_density_matrix({"kind": "3d"})


def test_plot_sky_backprojection_returns_figure() -> None:
    """Return a Figure from plot_sky_backprojection for a single image."""
    image = torch.zeros(4, 8)
    image[1, 2] = 5.0
    fig = plot_sky_backprojection(image, 10)
    assert isinstance(fig, Figure)
