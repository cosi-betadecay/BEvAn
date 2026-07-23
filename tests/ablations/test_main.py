import csv
import sys
from pathlib import Path

import harness
import main
import pytest


def test_run_comparison_rejects_unknown_name(tiny_split: tuple[dict, dict]) -> None:
    """Verify run_comparison raises on an unregistered ablation name."""
    train, eval_data = tiny_split
    with pytest.raises(ValueError, match="Unknown comparison ablation"):
        main.run_comparison("mystery", {}, train, eval_data, {}, {}, None, {}, plots_dir=None)


def test_main_raises_without_matching_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main raises when no discovered dataset matches the request."""
    monkeypatch.setattr(harness, "discover_datasets", list)
    with pytest.raises(RuntimeError, match="No matching dataset"):
        main.main(False, None, None, ())


def test_main_end_to_end_writes_figures_and_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_ds: harness.DatasetPaths,
    tiny_cache: Path,
) -> None:
    """Verify an end-to-end main run writes every expected table, figure, and confusion plot."""
    monkeypatch.setattr(harness, "discover_datasets", lambda: [tiny_ds])
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("label_window.N_STD_VALUES", (1, 5))  # keep the sweep short

    main.main(False, None, None, (), n_seeds=1)

    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    tables_dir = run_dirs[0] / "tables"
    figures_dir = run_dirs[0] / "figures"

    for name in (
        "learned_weights.csv",
        "no_ckd_order.csv",
        "summary.csv",
        "factor_contributions.csv",
        "label_window.csv",
        "dedicated_prior.csv",
    ):
        assert (tables_dir / name).is_file(), name
    for name in ("learned_weights.png", "no_ckd_order.png", "factor_contributions.png", "label_window.png"):
        assert (figures_dir / name).is_file(), name
    for variant in ("reference", "learned_weights", "no_ckd_order", "dedicated_prior"):
        assert (figures_dir / "tiny" / variant / "confusion_matrix_eval.png").is_file(), variant

    with (tables_dir / "dedicated_prior.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["dataset"], r["split"]) for r in rows] == [("tiny", "train"), ("tiny", "eval")]
    assert float(rows[0]["prior_bucket_1"]) == pytest.approx(0.5)


def test_main_multi_seed_writes_std_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_ds: harness.DatasetPaths,
    tiny_cache: Path,
) -> None:
    """Verify a multi-seed run adds finite std columns to the per-ablation tables."""
    monkeypatch.setattr(harness, "discover_datasets", lambda: [tiny_ds])
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("label_window.N_STD_VALUES", (5,))  # keep the sweep short

    main.main(False, None, None, (), n_seeds=2)

    tables_dir = next((tmp_path / "results").iterdir()) / "tables"
    with (tables_dir / "label_window.csv").open() as fh:
        label_rows = list(csv.DictReader(fh))
    for row in label_rows:
        for column in ("f1_score_std", "auc_std", "precision_std", "recall_std"):
            assert float(row[column]) >= 0.0  # finite spread from the two seeds
    with (tables_dir / "summary.csv").open() as fh:
        summary_rows = list(csv.DictReader(fh))
    assert all(float(r["delta_f1_score_std"]) >= 0.0 for r in summary_rows)
    with (tables_dir / "dedicated_prior.csv").open() as fh:
        dedicated_prior_rows = list(csv.DictReader(fh))
    assert all(float(r["f1_score_std"]) >= 0.0 for r in dedicated_prior_rows)


def test_main_respects_ablation_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_ds: harness.DatasetPaths,
    tiny_cache: Path,
) -> None:
    """Verify an ablation filter runs only the named ablation and skips the rest."""
    monkeypatch.setattr(harness, "discover_datasets", lambda: [tiny_ds])
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")

    main.main(False, None, ["dedicated_prior"], (), n_seeds=1)

    tables_dir = next((tmp_path / "results").iterdir()) / "tables"
    assert (tables_dir / "dedicated_prior.csv").is_file()
    assert not (tables_dir / "summary.csv").exists()
    assert not (tables_dir / "label_window.csv").exists()


def test_parse_args_defaults_and_choices(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify parse_args supplies the documented defaults and rejects unknown ablation choices."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = main.parse_args()
    assert not args.wandb
    assert args.datasets is None
    assert args.ablations is None
    assert args.exclude_avg == []

    monkeypatch.setattr(sys, "argv", ["main.py", "--ablations", "not_an_ablation"])
    with pytest.raises(SystemExit):
        main.parse_args()
    capsys.readouterr()
