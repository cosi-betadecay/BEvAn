import dedicated_prior
import pytest


def test_run_deploys_at_the_dedicated_prior_point(tiny_ds, tiny_cache, tiny_split, tmp_path):
    train, eval_data = tiny_split
    out = tmp_path / "dedicated_prior"
    record = dedicated_prior.run(tiny_ds, train, eval_data, plots_dir=out)

    # The synthetic prior file holds 3/6/6 events per class in buckets 1/2/3.
    assert record["prior_counts"] == {
        1: {"bdecay": 3, "bg": 3},
        2: {"bdecay": 6, "bg": 6},
        3: {"bdecay": 6, "bg": 6},
    }
    assert record["priors"] == {1: pytest.approx(0.5), 2: pytest.approx(0.5), 3: pytest.approx(0.5)}
    for split in ("train", "eval"):
        assert {"tp", "fp", "fn", "tn", "f1_score", "best_f1", "auc"} <= set(record[split])

    assert (out / "confusion_matrix_eval.png").is_file()
    assert (out / "sky_backprojection.png").is_file()
