import sys
from pathlib import Path

import pytest
import torch
import train_model

from pipeline.train import Trainer


def test_train_and_save_validates_inputs(tmp_path: Path, synthetic_dataset: dict) -> None:
    """Reject wrong extensions, a missing sim file, and an empty prior list."""
    ds = synthetic_dataset
    out = str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="sim-file"):
        train_model.train_and_save(ds["geo"], ds["tra"], ds["tra"], out, False, [ds["prior"]])
    with pytest.raises(ValueError, match="tra-file"):
        train_model.train_and_save(ds["geo"], ds["sim"], ds["sim"], out, False, [ds["prior"]])
    with pytest.raises(FileNotFoundError, match="Simulation"):
        train_model.train_and_save(ds["geo"], str(tmp_path / "no.sim"), ds["tra"], out, False, [ds["prior"]])
    with pytest.raises(ValueError, match="prior"):
        train_model.train_and_save(ds["geo"], ds["sim"], ds["tra"], out, False, [])


def test_train_and_save_end_to_end(tmp_path: Path, synthetic_dataset: dict) -> None:
    """Train and save a model end to end with the expected counts and provenance."""
    ds = synthetic_dataset
    out = tmp_path / "models" / "tiny.pt"
    train_model.train_and_save(ds["geo"], ds["sim"], ds["tra"], str(out), False, [ds["prior"]])
    assert out.is_file()

    trainer = Trainer.load(out)
    # Fit on the full simulation with the counted prior applied: the synthetic
    # dataset has 3/6/6 events per class in buckets 1/2/3, in both roles.
    for bucket, n in ((1, 3), (2, 6), (3, 6)):
        assert trainer.models[bucket]["n_beta"] == n
        assert trainer.models[bucket]["n_bg"] == n
        assert trainer.models[bucket]["terms"]

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["provenance"]["sim_file"] == ds["sim"]
    assert payload["provenance"]["prior_sim_files"] == [ds["prior"]]
    assert payload["provenance"]["prior_counts"][2] == {"bdecay": 6, "bg": 6}


def test_parse_args_requires_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require the path args and parse a valid argv into the namespace."""
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--sim-file", "a.sim"])
    with pytest.raises(SystemExit):
        train_model.parse_args()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--geo-file",
            "g.setup",
            "--sim-file",
            "a.sim",
            "--tra-file",
            "a.tra",
            "--prior-sim",
            "p.sim",
        ],
    )
    args = train_model.parse_args()
    assert args.out is None
    assert args.geo_file == "g.setup"
