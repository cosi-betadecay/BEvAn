import json
import sys

import analysis
import pytest


def test_main_validates_file_extensions(synthetic_dataset):
    ds = synthetic_dataset
    with pytest.raises(ValueError, match="sim-file"):
        analysis.main(ds["geo"], ds["tra"], ds["tra"], False, [ds["prior"]])
    with pytest.raises(ValueError, match="tra-file"):
        analysis.main(ds["geo"], ds["sim"], ds["sim"], False, [ds["prior"]])


def test_main_validates_file_existence(tmp_path, synthetic_dataset):
    ds = synthetic_dataset
    with pytest.raises(FileNotFoundError, match="Simulation"):
        analysis.main(ds["geo"], str(tmp_path / "missing.sim"), ds["tra"], False, [ds["prior"]])
    with pytest.raises(FileNotFoundError, match="Tra"):
        analysis.main(ds["geo"], ds["sim"], str(tmp_path / "missing.tra"), False, [ds["prior"]])
    with pytest.raises(ValueError, match="prior"):
        analysis.main(ds["geo"], ds["sim"], ds["tra"], False, [])


def test_main_end_to_end_writes_results(monkeypatch, tmp_path, synthetic_dataset):
    ds = synthetic_dataset
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path / "results")
    analysis.main(ds["geo"], ds["sim"], ds["tra"], False, [ds["prior"]])

    out_dirs = list((tmp_path / "results").glob("*/tiny"))
    assert len(out_dirs) == 1
    out = out_dirs[0]

    record = json.loads((out / "metrics.json").read_text())
    assert record["dataset"] == "tiny"
    assert record["selection"]["prior_sim_files"] == [ds["prior"]]
    for split in ("train", "eval"):
        block = record[split]
        assert {"tp", "fp", "fn", "tn", "excluded", "f1_score", "best_f1", "auc"} <= set(block)
        assert block["tp"] + block["fp"] + block["fn"] + block["tn"] > 0

    for name in (
        "confusion_matrix_train.png",
        "confusion_matrix_eval.png",
        "roc_eval.png",
        "pr_eval.png",
        "sky_backprojection.png",
        "density_n1_delta_E.png",
    ):
        assert (out / name).is_file(), name


def test_parse_args_requires_sim_tra_and_prior(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["analysis.py"])
    with pytest.raises(SystemExit):
        analysis.parse_args()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["analysis.py", "--sim-file", "a.sim", "--tra-file", "a.tra", "--prior-sim", "p1.sim", "p2.sim"],
    )
    args = analysis.parse_args()
    assert args.sim_file == "a.sim"
    assert args.prior_sim == ["p1.sim", "p2.sim"]
    assert not args.wandb
