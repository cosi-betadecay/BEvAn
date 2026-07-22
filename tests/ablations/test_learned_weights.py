import harness
import learned_weights


def test_run_returns_metrics_and_fitted_weights(tiny_split, tmp_path):
    train, eval_data = tiny_split
    out = tmp_path / "learned_weights"
    record = learned_weights.run(train, eval_data, harness.Trainer().config(), plots_dir=out)

    expected = {"tp", "fp", "fn", "tn", "excluded", "f1_score", "best_f1", "auc", "bias"}
    assert expected <= set(record)
    assert {f"w_{factor}" for factor in harness.FACTORS} <= set(record)
    assert all(isinstance(record[f"w_{factor}"], float) for factor in harness.FACTORS)
    assert isinstance(record["bias"], float)
    assert record["excluded"] == 0  # scored events are pre-filtered to finite features

    assert (out / "confusion_matrix_eval.png").is_file()
    assert (out / "roc_eval.png").is_file()
    assert (out / "pr_eval.png").is_file()
    assert list(out.glob("density_*.png"))


def test_run_is_deterministic(tiny_split):
    train, eval_data = tiny_split
    config = harness.Trainer().config()
    first = learned_weights.run(train, eval_data, config)
    second = learned_weights.run(train, eval_data, config)
    assert [first[f"w_{f}"] for f in harness.FACTORS] == [second[f"w_{f}"] for f in harness.FACTORS]
    assert first["f1_score"] == second["f1_score"]
