from pathlib import Path

import pytest
import torch

from pipeline.datasets import BUCKETS, FEATURES
from pipeline.train import REFERENCE_BINS, Trainer, flatten_config


def models_equal(a: dict[int, dict], b: dict[int, dict]) -> bool:
    """Whether two per-bucket model dicts hold identical counts and term tensors."""
    for bucket in BUCKETS:
        ma, mb = a[bucket], b[bucket]
        if (ma["n_beta"], ma["n_bg"], len(ma["terms"])) != (mb["n_beta"], mb["n_bg"], len(mb["terms"])):
            return False
        for ta, tb in zip(ma["terms"], mb["terms"], strict=True):
            if any(
                (torch.is_tensor(ta[k]) and not torch.equal(ta[k], tb[k]))
                or (not torch.is_tensor(ta[k]) and ta[k] != tb[k])
                for k in ta
            ):
                return False
    return True


def test_trainer_defaults_to_reference_bins() -> None:
    """Default the trainer to the reference bin counts, honoring an override."""
    assert Trainer().n_bins == REFERENCE_BINS
    assert Trainer(n_bins={1: 5, 2: 5, 3: 5}).n_bins == {1: 5, 2: 5, 3: 5}


def test_flatten_config_scalar_columns() -> None:
    """Flatten a nested config into scalar float columns for logging."""
    flat = flatten_config({"n_bins": {1: 25, 2: 8, 3: 35}, "joint_smoothing": 0.5, "marginal_smoothing": 0.0})
    assert flat == {
        "n_bins_1": 25.0,
        "n_bins_2": 8.0,
        "n_bins_3": 35.0,
        "joint_smoothing": 0.5,
        "marginal_smoothing": 0.0,
    }


def test_fit_builds_expected_terms_per_bucket(
    separable_data: dict[str, dict[int, dict[str, torch.Tensor]]],
) -> None:
    """Build the expected density terms and class counts for each bucket."""
    trainer = Trainer().fit(separable_data)
    kinds = {b: [(t["kind"], t.get("yfeat")) for t in trainer.models[b]["terms"]] for b in BUCKETS}
    assert kinds[1] == [("1d", None)]
    assert kinds[2] == [("2d", "arm")]
    assert kinds[3] == [("2d", "arm"), ("2d", "anni")]
    n = separable_data["bdecay"][1]["delta_E"].numel()
    for b in BUCKETS:
        assert trainer.models[b]["n_beta"] == n
        assert trainer.models[b]["n_bg"] == n


def test_fit_empty_class_yields_prior_only_model(
    separable_data: dict[str, dict[int, dict[str, torch.Tensor]]],
) -> None:
    """Yield a prior-only model with no terms when a class bucket is empty."""
    for f in FEATURES:
        separable_data["bdecay"][2][f] = torch.tensor([])
    model = Trainer().fit(separable_data).models[2]
    assert model["terms"] == []
    assert model["n_beta"] == 0


def test_save_load_roundtrip(
    tmp_path: Path, separable_data: dict[str, dict[int, dict[str, torch.Tensor]]]
) -> None:
    """Save a fitted trainer and load back identical config, models, and provenance."""
    trainer = Trainer(n_bins={1: 10, 2: 10, 3: 10}, joint_smoothing=0.7).fit(separable_data)
    path = tmp_path / "model.pt"
    trainer.save(path, provenance={"sim": "synthetic"})

    loaded = Trainer.load(path)
    assert loaded.config() == trainer.config()
    assert models_equal(loaded.models, trainer.models)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["provenance"] == {"sim": "synthetic"}


def test_save_requires_fit() -> None:
    """Refuse to save a trainer that has not been fitted."""
    with pytest.raises(RuntimeError, match="fit"):
        Trainer().save("never_written.pt")


def test_load_rejects_stale_artifacts(
    tmp_path: Path, separable_data: dict[str, dict[int, dict[str, torch.Tensor]]]
) -> None:
    """Reject artifacts with a stale format version or feature signature."""
    trainer = Trainer().fit(separable_data)
    path = tmp_path / "model.pt"
    trainer.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    bad_version = dict(payload, format_version=999)
    torch.save(bad_version, tmp_path / "bad_version.pt")
    with pytest.raises(ValueError, match="format version"):
        Trainer.load(tmp_path / "bad_version.pt")

    bad_signature = dict(payload, feature_signature={"features": ["other"], "buckets": [1]})
    torch.save(bad_signature, tmp_path / "bad_signature.pt")
    with pytest.raises(ValueError, match="signature"):
        Trainer.load(tmp_path / "bad_signature.pt")
