import pytest
import torch
import wandb

from utils.wandb_logging import (
    log_confusion_matrix,
    log_density_matrix,
    log_density_terms,
    log_pr_curve,
    log_roc_curve,
    log_score_curves,
    log_sky_backprojection,
)

COUNTS = {"tp": 5, "fp": 1, "fn": 2, "tn": 8}
SCORES = (torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0]))
TERM_1D = {
    "kind": "1d",
    "xfeat": "delta_E",
    "x_bins": torch.tensor([0.1, 1.0, 10.0]),
    "beta": torch.tensor([0.75, 0.25]),
    "bg": torch.tensor([0.25, 0.75]),
}


class TrainerStub:
    models = {1: {"terms": [TERM_1D]}, 2: {"terms": []}}  # noqa: RUF012


@pytest.fixture()
def logged(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake an active W&B run and capture everything the helpers log."""
    captured: dict = {}
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", captured.update)
    monkeypatch.setattr(wandb, "Image", lambda fig: "image")
    return captured


def test_helpers_noop_without_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log nothing from any helper when there is no active W&B run."""
    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "log", pytest.fail)
    log_confusion_matrix(COUNTS)
    log_roc_curve(*SCORES)
    log_pr_curve(*SCORES)
    log_density_matrix(TERM_1D)
    log_score_curves(SCORES)
    log_sky_backprojection(torch.zeros(2, 2), 1, (0.0, 6.28), (0.0, 3.14))
    log_density_terms(TrainerStub())


def test_log_confusion_matrix_key(logged: dict) -> None:
    """Log the confusion matrix under the split-namespaced key."""
    log_confusion_matrix(COUNTS, split_name="eval/n2")
    assert set(logged) == {"eval/n2/confusion_matrix"}


def test_log_score_curves_keys(logged: dict) -> None:
    """Log ROC and PR curves under the split-namespaced keys."""
    log_score_curves(SCORES, split_name="train")
    assert set(logged) == {"train/roc_curve", "train/pr_curve"}


def test_log_score_curves_none_is_noop(logged: dict) -> None:
    """Log nothing when there are no pooled scores."""
    log_score_curves(None)
    assert logged == {}


def test_log_density_terms_namespaced_by_bucket(logged: dict) -> None:
    """Log each density term under a bucket-namespaced key."""
    log_density_terms(TrainerStub())
    assert set(logged) == {"model/density/n1/delta_E"}


def test_log_sky_backprojection_key(logged: dict) -> None:
    """Log the sky backprojection under its model-namespaced key."""
    log_sky_backprojection(torch.zeros(2, 2), 1, (0.0, 6.28), (0.0, 3.14))
    assert set(logged) == {"model/sky_backprojection"}
