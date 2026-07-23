import sys
from pathlib import Path

import inference
import pytest
import torch
import train_model


@pytest.fixture()
def model_file(tmp_path: Path, synthetic_dataset: dict) -> str:
    """Train and save a tiny model on the synthetic dataset, returning its path."""
    ds = synthetic_dataset
    out = tmp_path / "tiny_model.pt"
    train_model.train_and_save(ds["geo"], ds["sim"], ds["tra"], str(out), False, [ds["prior"]])
    return str(out)


def test_run_inference_validates_inputs(
    tmp_path: Path, synthetic_dataset: dict, model_file: str
) -> None:
    """Reject wrong extensions and a missing model before running inference."""
    ds = synthetic_dataset
    out = str(tmp_path / "classifications.pt")
    with pytest.raises(ValueError, match="sim-file"):
        inference.run_inference(model_file, ds["geo"], ds["tra"], ds["tra"], out)
    with pytest.raises(ValueError, match="tra-file"):
        inference.run_inference(model_file, ds["geo"], ds["sim"], ds["sim"], out)
    with pytest.raises(FileNotFoundError, match="Model"):
        inference.run_inference(str(tmp_path / "no.pt"), ds["geo"], ds["sim"], ds["tra"], out)


def test_run_inference_end_to_end(
    tmp_path: Path, synthetic_dataset: dict, model_file: str
) -> None:
    """Run inference end to end and classify the separable events exactly."""
    ds = synthetic_dataset
    out = tmp_path / "results" / "classifications.pt"
    classifications = inference.run_inference(model_file, ds["geo"], ds["sim"], ds["tra"], str(out))

    assert classifications.dtype == torch.uint8
    assert classifications.numel() == ds["n_nonempty"]  # one entry per non-empty event, file order
    assert set(classifications.tolist()) <= {0, 1}
    assert torch.equal(torch.load(out, weights_only=False), classifications)

    # The synthetic events alternate beta (odd ids) / background (even ids); the
    # model was fit on this same data, so at least the separable buckets 1 and 2
    # (events 1..18) must be classified exactly.
    expected = torch.tensor([1, 0] * 9, dtype=torch.uint8)
    assert torch.equal(classifications[:18], expected)


def test_parse_args_requires_model_and_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require model and path args and parse a valid argv into the namespace."""
    monkeypatch.setattr(sys, "argv", ["inference.py", "--sim-file", "a.sim"])
    with pytest.raises(SystemExit):
        inference.parse_args()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--model",
            "m.pt",
            "--geo-file",
            "g.setup",
            "--sim-file",
            "a.sim",
            "--tra-file",
            "a.tra",
        ],
    )
    args = inference.parse_args()
    assert args.model == "m.pt"
    assert args.out is None
