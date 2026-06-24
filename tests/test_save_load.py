"""Round-trip tests for the Trainer save/load seam.

The trained model is just per-bucket density matrices plus scalars, so saving
and reloading it must reproduce identical inference. These tests use synthetic
feature tensors (no MEGAlib reader), but importing the pipeline transitively
pulls in PyROOT via dataset.datasets, so the whole module is skipped where ROOT
is unavailable.
"""

import pytest

pytest.importorskip("ROOT")  # the pipeline transitively imports PyROOT

import torch

from dataset.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.eval import Evaluator
from pipeline.model_selection import apply_offset
from pipeline.train import SAVE_FORMAT_VERSION, Trainer


def _synthetic_data(n: int = 300, seed: int = 0) -> dict:
    """Build a nested per-class, per-bucket feature dict of finite positive values.

    Positive values keep the log-spaced delta_E/ARM edges well-defined; the two
    classes are given slightly different scales so the densities actually separate.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = {"bdecay": 1.0, "bg": 1.6}
    data = {}
    for cls in CLASSES:
        data[cls] = {}
        for b in BUCKETS:
            data[cls][b] = {f: (torch.rand(n, generator=generator) + 0.05) * scale[cls] for f in FEATURES}
    return data


def test_save_load_reproduces_confusion(tmp_path):
    """Fit -> save -> load -> evaluate must match fit -> evaluate exactly."""
    data = _synthetic_data()
    trainer = Trainer(rho_floor=10.0).fit(data)
    before = Evaluator(trainer).evaluate(data, split_name="orig")

    path = tmp_path / "model.pt"
    trainer.save(path)
    loaded = Trainer.load(path)
    after = Evaluator(loaded).evaluate(data, split_name="loaded")

    assert before == after


def test_calibrated_threshold_survives_round_trip(tmp_path):
    """A calibrated log_threshold baked in before save is restored on load."""
    data = _synthetic_data()
    trainer = Trainer(rho_floor=10.0).fit(data)
    apply_offset(trainer, offset=0.5)
    before = Evaluator(trainer).evaluate(data, split_name="orig")

    path = tmp_path / "model.pt"
    trainer.save(path, threshold_offset=0.5)
    loaded = Trainer.load(path)

    for b in BUCKETS:
        assert loaded.models[b].get("log_threshold") == trainer.models[b].get("log_threshold")
    assert Evaluator(loaded).evaluate(data, split_name="loaded") == before


def test_load_rejects_bad_version(tmp_path):
    """A payload with the wrong format version is refused with a clear error."""
    data = _synthetic_data()
    path = tmp_path / "model.pt"
    Trainer(rho_floor=10.0).fit(data).save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = SAVE_FORMAT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        Trainer.load(path)


def test_load_rejects_bad_signature(tmp_path):
    """A payload whose feature signature differs is refused on load."""
    data = _synthetic_data()
    path = tmp_path / "model.pt"
    Trainer(rho_floor=10.0).fit(data).save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["feature_signature"] = {"features": ["delta_E"], "buckets": [1]}
    torch.save(payload, path)

    with pytest.raises(ValueError, match="feature signature"):
        Trainer.load(path)
