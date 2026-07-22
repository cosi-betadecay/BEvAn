import dedicated_prior
import harness
import label_window
import pytest


def test_default_sweep_covers_the_deployed_window():
    assert tuple(range(1, 11)) == label_window.N_STD_VALUES
    assert 5 in label_window.N_STD_VALUES


def test_run_relabels_per_window(tiny_ds, tiny_cache):
    results = label_window.run(tiny_ds, n_std_values=(1, 5))
    assert set(results) == {1, 5}
    for record in results.values():
        assert {"tp", "fp", "fn", "tn", "f1_score", "best_f1", "auc"} <= set(record)
    # The synthetic beta events sit within ~3.4 keV of 511, inside the 5-sigma
    # (~4.78 keV) window but outside the 1-sigma (~0.96 keV) one for most events,
    # so shrinking the window must move events from beta to background.
    positives = {n: rec["tp"] + rec["fn"] for n, rec in results.items()}
    assert positives[1] < positives[5]


def test_deployed_window_reproduces_the_dedicated_prior_record(tiny_ds, tiny_cache, tiny_split):
    """The whole point of the per-window prior: n_std=5 IS the deployed run.

    Same dataset, same seed, same dedicated-prior operating point — so the sweep's
    anchor must land on ``dedicated_prior``'s eval record, not a training-count one.
    """
    train, eval_data = tiny_split
    deployed = dedicated_prior.run(tiny_ds, train, eval_data, plots_dir=None)["eval"]
    swept = label_window.run(tiny_ds, n_std_values=(5,))[5]

    assert set(swept) == set(deployed)
    for key, value in deployed.items():
        assert swept[key] == pytest.approx(value, nan_ok=True), key


def test_sweep_uses_the_prior_pool_not_the_training_counts(tiny_ds, tiny_cache, monkeypatch):
    """A doctored prior must move the sweep's operating point, per window."""
    baseline = label_window.run(tiny_ds, n_std_values=(5,))[5]

    # A prior swung hard toward background raises every bucket's decision threshold,
    # so fewer events can be tagged β⁺ than at the pool's balanced prior.
    monkeypatch.setattr(
        label_window.harness,
        "prior_pool_counts",
        lambda pool, n_std=5: {b: {"bdecay": 1, "bg": 10_000} for b in harness.BUCKETS},
    )
    assert label_window.run(tiny_ds, n_std_values=(5,))[5]["tp"] < baseline["tp"]
