import math
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import plots
import pytest


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close every matplotlib figure a test opens once it finishes."""
    yield
    plt.close("all")


def record(f1: float) -> dict[str, float]:
    """Build a minimal metric record with the given F1 and placeholder metrics."""
    return {"f1_score": f1, "auc": 0.9, "precision": f1, "recall": f1}


def test_pretty_names() -> None:
    """Verify pretty maps ablation ids to human-readable, sentence-cased names."""
    assert plots.pretty("no_ckd_order") == "No CKD ordering"
    assert plots.pretty("some_new_thing") == "Some new thing"


def test_mean_std_skips_nan() -> None:
    """Verify mean_std ignores NaN values and reports zero std for a single value."""
    mean, std = plots.mean_std([1.0, 3.0, float("nan")])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)
    nan_mean, zero_std = plots.mean_std([float("nan")])
    assert math.isnan(nan_mean)
    assert zero_std == 0.0


def test_save_writes_and_closes(tmp_path: Path) -> None:
    """Verify save writes the figure (creating parents) and closes it afterwards."""
    fig = plt.figure()
    path = plots.save(fig, tmp_path / "nested" / "figure.png")
    assert path.is_file()
    assert fig.number not in plt.get_fignums()


def test_plot_metric_comparison(tmp_path: Path) -> None:
    """Verify plot_metric_comparison writes the comparison figure at the expected path."""
    per_dataset = {
        "A": {"our_model": record(0.9), "ablation": record(0.8)},
        "B": {"our_model": record(0.7), "ablation": record(0.6)},
    }
    path = plots.plot_metric_comparison("no_ckd_order", per_dataset, tmp_path, exclude=("B",))
    assert path == tmp_path / "no_ckd_order.png"
    assert path.is_file()


def test_plot_label_window(tmp_path: Path) -> None:
    """Verify plot_label_window writes the label-window sweep figure at the expected path."""
    results = {"A": {1: record(0.5), 5: record(0.9)}, "B": {1: record(0.6), 5: record(0.8)}}
    path = plots.plot_label_window(results, tmp_path)
    assert path == tmp_path / "label_window.png"
    assert path.is_file()


def test_plot_factor_contributions(tmp_path: Path) -> None:
    """Verify plot_factor_contributions writes the per-factor figure at the expected path."""
    contributions = {"A": {"delta_E": {"f1": 0.8, "auc": 0.9}, "arm": {"f1": 0.6, "auc": 0.7}}}
    full = {"A": {"f1": 0.95, "auc": 0.99}}
    path = plots.plot_factor_contributions(contributions, full, tmp_path)
    assert path == tmp_path / "factor_contributions.png"
    assert path.is_file()
