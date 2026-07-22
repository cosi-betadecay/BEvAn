import json

import matplotlib.pyplot as plt
import pytest
import torch

from utils.local_results import (
    save_confusion_matrix,
    save_density_terms,
    save_figure,
    save_score_curves,
    save_sky_backprojection,
    write_metrics_json,
)

COUNTS = {"tp": 5, "fp": 1, "fn": 2, "tn": 8}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_save_figure_creates_parents_and_closes(tmp_path):
    fig = plt.figure()
    path = tmp_path / "nested" / "figure.png"
    save_figure(fig, path)
    assert path.is_file()
    assert fig.number not in plt.get_fignums()


def test_save_confusion_matrix(tmp_path):
    save_confusion_matrix(COUNTS, tmp_path, "train")
    assert (tmp_path / "confusion_matrix_train.png").is_file()


def test_save_score_curves(tmp_path):
    pooled = (torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0]))
    save_score_curves(pooled, tmp_path, "eval")
    assert (tmp_path / "roc_eval.png").is_file()
    assert (tmp_path / "pr_eval.png").is_file()


def test_save_score_curves_none_is_noop(tmp_path):
    save_score_curves(None, tmp_path, "eval")
    assert list(tmp_path.iterdir()) == []


def test_save_density_terms_one_file_per_term(tmp_path):
    term = {
        "kind": "1d",
        "xfeat": "delta_E",
        "x_bins": torch.tensor([0.1, 1.0, 10.0]),
        "beta": torch.tensor([0.75, 0.25]),
        "bg": torch.tensor([0.25, 0.75]),
    }

    class TrainerStub:
        models = {1: {"terms": [term]}, 2: {"terms": []}}  # noqa: RUF012

    save_density_terms(TrainerStub(), tmp_path)
    assert (tmp_path / "density_n1_delta_E.png").is_file()
    assert len(list(tmp_path.glob("density_*.png"))) == 1


def test_save_sky_backprojection(tmp_path):
    image = torch.zeros(4, 8)
    image[2, 3] = 1.0
    save_sky_backprojection([("All events", image, 7)], tmp_path, (0.0, 6.28), (0.0, 3.14))
    assert (tmp_path / "sky_backprojection.png").is_file()


def test_write_metrics_json_roundtrip(tmp_path):
    record = {"dataset": "tiny", "train": COUNTS}
    path = write_metrics_json(tmp_path / "run", record)
    assert path == tmp_path / "run" / "metrics.json"
    assert json.loads(path.read_text()) == record
