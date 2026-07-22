import csv
import math

import pytest
import tables


def record(f1: float, auc: float = 0.9, fpr: float = 0.1) -> dict:
    return {"f1_score": f1, "auc": auc, "precision": f1, "recall": f1, "fpr": fpr}


COMPARISON = {
    "A": {"our_model": record(0.9), "ablation": record(0.8)},
    "B": {"our_model": record(0.7), "ablation": record(0.6)},
}


def read_csv(path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_mean_skips_nan():
    assert tables.mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert math.isnan(tables.mean([float("nan")]))
    assert math.isnan(tables.mean([]))


def test_save_comparison_csv_rows_and_mean(tmp_path):
    path = tables.save_comparison_csv("no_ckd_order", COMPARISON, tmp_path)
    assert path == tmp_path / "no_ckd_order.csv"
    rows = read_csv(path)
    assert [r["dataset"] for r in rows] == ["A", "B", "mean"]
    assert float(rows[0]["our_model_f1_score"]) == pytest.approx(0.9)
    assert float(rows[0]["ablation_f1_score"]) == pytest.approx(0.8)
    assert float(rows[2]["our_model_f1_score"]) == pytest.approx(0.8)
    assert float(rows[0]["our_model_fpr"]) == pytest.approx(0.1)


def test_save_comparison_csv_exclude_only_affects_mean(tmp_path):
    rows = read_csv(tables.save_comparison_csv("x", COMPARISON, tmp_path, exclude=("B",)))
    assert [r["dataset"] for r in rows] == ["A", "B", "mean (excl. B)"]
    assert float(rows[2]["our_model_f1_score"]) == pytest.approx(0.9)  # B dropped from the mean only


def test_summary_row_deltas():
    row = tables.summary_row("no_ckd_order", COMPARISON)
    assert row["ablation"] == "no_ckd_order"
    assert row["our_model_f1_score"] == pytest.approx(0.8)
    assert row["ablation_f1_score"] == pytest.approx(0.7)
    assert row["delta_f1_score"] == pytest.approx(-0.1)
    assert math.isnan(row["delta_f1_score_std"])  # no per-seed pairs given


def test_summary_row_paired_delta_std():
    per_seed = {
        "A": [
            {"our_model": record(0.9), "ablation": record(0.8)},  # delta -0.10
            {"our_model": record(0.7), "ablation": record(0.65)},  # delta -0.05
        ]
    }
    aggregated = {"A": {"our_model": record(0.8), "ablation": record(0.725)}}
    row = tables.summary_row("x", aggregated, per_dataset_seeds=per_seed)
    assert row["delta_f1_score"] == pytest.approx(-0.075)
    # Std across seeds of the paired per-seed deltas [-0.10, -0.05].
    assert row["delta_f1_score_std"] == pytest.approx(0.025)


def test_save_summary_csv(tmp_path):
    rows = [tables.summary_row("x", COMPARISON)]
    csv_rows = read_csv(tables.save_summary_csv(rows, tmp_path))
    assert len(csv_rows) == 1
    assert csv_rows[0]["ablation"] == "x"
    assert float(csv_rows[0]["delta_auc"]) == pytest.approx(0.0)


def test_save_factor_contributions_csv_long_format(tmp_path):
    contributions = {"A": {"delta_E": {"f1": 0.8, "auc": 0.9}, "arm": {"f1": 0.6, "auc": 0.7}}}
    full = {"A": {"f1": 0.95, "auc": 0.99}}
    rows = read_csv(tables.save_factor_contributions_csv(contributions, full, tmp_path))
    assert [(r["dataset"], r["factor"]) for r in rows] == [("A", "delta_E"), ("A", "arm")]
    assert float(rows[0]["full_f1"]) == pytest.approx(0.95)
    assert float(rows[1]["auc"]) == pytest.approx(0.7)


def test_save_label_window_csv_with_mean_rows(tmp_path):
    results = {
        "A": {1: record(0.5), 5: record(0.9)},
        "B": {1: record(0.7), 5: record(0.7)},
    }
    rows = read_csv(tables.save_label_window_csv(results, {}, tmp_path, exclude=("B",)))
    assert [(r["dataset"], r["n_std"]) for r in rows] == [
        ("A", "1"),
        ("A", "5"),
        ("B", "1"),
        ("B", "5"),
        ("mean (excl. B)", "1"),
        ("mean (excl. B)", "5"),
    ]
    assert float(rows[4]["f1_score"]) == pytest.approx(0.5)
    assert float(rows[5]["f1_score"]) == pytest.approx(0.9)


def test_save_label_window_csv_std_columns(tmp_path):
    seeded = {
        **record(0.5),
        "f1_score_std": 0.02,
        "auc_std": 0.01,
        "precision_std": 0.03,
        "recall_std": 0.04,
        "fpr_std": 0.05,
    }
    rows = read_csv(tables.save_label_window_csv({"A": {1: seeded}}, {}, tmp_path))
    for column, value in (
        ("f1_score_std", 0.02),
        ("auc_std", 0.01),
        ("precision_std", 0.03),
        ("recall_std", 0.04),
        ("fpr_std", 0.05),
    ):
        assert float(rows[0][column]) == pytest.approx(value)
        assert float(rows[1][column]) == pytest.approx(value)  # mean row: mean of per-dataset spreads


def hrecord(f: float, fpr: float = 0.01) -> dict:
    return {"f1_score": f, "precision": f, "recall": f, "fpr": fpr, "tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_save_label_window_csv_has_heuristic_columns(tmp_path):
    results = {"A": {1: record(0.5), 5: record(0.9)}, "B": {1: record(0.7), 5: record(0.7)}}
    heuristic = {"A": {1: hrecord(0.2), 5: hrecord(0.3)}, "B": {1: hrecord(0.25), 5: hrecord(0.28)}}
    path = tables.save_label_window_csv(results, heuristic, tmp_path, exclude=("B",))
    with path.open() as fh:
        header = next(csv.reader(fh))
    assert header[-4:] == ["heuristic_f1_score", "heuristic_precision", "heuristic_recall", "heuristic_fpr"]
    assert "heuristic_auc" not in header and "heuristic_f1_score_std" not in header  # no auc, no spread
    rows = read_csv(path)
    assert float(rows[0]["heuristic_f1_score"]) == pytest.approx(0.2)  # A, n_std=1
    # mean row (excl. B) averages only A -> 0.3 at n_std=5
    assert float(rows[-1]["heuristic_recall"]) == pytest.approx(0.3)


def test_save_label_window_csv_heuristic_columns_always_present(tmp_path):
    # The baseline is a fixed part of this table: columns exist even with no data,
    # holding NaN rather than being dropped.
    path = tables.save_label_window_csv({"A": {1: record(0.5)}}, {}, tmp_path)
    rows = read_csv(path)
    assert {"heuristic_f1_score", "heuristic_precision", "heuristic_recall", "heuristic_fpr"} <= set(rows[0])
    assert math.isnan(float(rows[0]["heuristic_f1_score"]))


def test_save_dedicated_prior_csv_std_columns(tmp_path):
    results = {
        "A": {
            "priors": {1: 0.5},
            "train": {**record(0.9), "tp": 9, "best_f1": 0.92, "f1_score_std": 0.015, "best_f1_std": 0.01},
            "eval": {**record(0.8), "tp": 4, "best_f1": 0.85, "f1_score_std": 0.025, "best_f1_std": 0.02},
        }
    }
    rows = read_csv(tables.save_dedicated_prior_csv(results, tmp_path))
    assert float(rows[0]["f1_score_std"]) == pytest.approx(0.015)
    assert float(rows[1]["f1_score_std"]) == pytest.approx(0.025)
    assert float(rows[1]["best_f1_std"]) == pytest.approx(0.02)


def test_save_dedicated_prior_csv_per_split_rows(tmp_path):
    results = {
        "A": {
            "priors": {1: 0.5, 2: 0.25, 3: 0.1},
            "train": {**record(0.9), "tp": 9, "fp": 1, "fn": 1, "tn": 9, "excluded": 0, "best_f1": 0.92},
            "eval": {**record(0.8), "tp": 4, "fp": 1, "fn": 1, "tn": 4, "excluded": 1, "best_f1": 0.85},
        }
    }
    rows = read_csv(tables.save_dedicated_prior_csv(results, tmp_path))
    assert [(r["dataset"], r["split"]) for r in rows] == [("A", "train"), ("A", "eval")]
    assert rows[0]["tp"] == "9"
    assert rows[1]["excluded"] == "1"
    assert float(rows[0]["prior_bucket_2"]) == pytest.approx(0.25)
    assert float(rows[1]["best_f1"]) == pytest.approx(0.85)


def test_save_comparison_csv_writes_extra_ablation_columns(tmp_path):
    """learned_weights persists its fitted weights; other ablations gain no columns."""
    weighted = {
        "A": {"our_model": record(0.9), "ablation": record(0.8) | {"w_delta_E": 0.5, "bias": -1.8}},
        "B": {"our_model": record(0.7), "ablation": record(0.6) | {"w_delta_E": 0.7, "bias": -2.0}},
    }
    rows = read_csv(tables.save_comparison_csv("learned_weights", weighted, tmp_path))
    assert float(rows[0]["ablation_w_delta_E"]) == pytest.approx(0.5)
    assert float(rows[2]["ablation_w_delta_E"]) == pytest.approx(0.6)  # mean row
    assert float(rows[2]["ablation_bias"]) == pytest.approx(-1.9)
    assert math.isnan(float(rows[0]["ablation_w_arm"]))  # absent from the record -> NaN, not a crash
    assert math.isnan(float(rows[0]["ablation_w_delta_E_std"]))  # single-seed: no spread

    plain = read_csv(tables.save_comparison_csv("no_ckd_order", COMPARISON, tmp_path))
    assert not [c for c in plain[0] if c.startswith("ablation_w_")]
