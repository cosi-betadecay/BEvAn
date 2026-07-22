import harness
import no_ckd_order


def test_run_fits_reference_config_on_energy_features(tiny_ds, tiny_cache):
    train_energy, eval_energy = harness.extract_split(tiny_ds, ordering="energy")
    assert (tiny_cache / "tiny_energy.pt").is_file()

    counts = harness.dedicated_prior_counts(tiny_ds)
    record = no_ckd_order.run(train_energy, eval_energy, harness.Trainer().config(), counts)
    expected = {"tp", "fp", "fn", "tn", "excluded", "f1_score", "best_f1", "auc"}
    assert expected <= set(record)
    n_eval = sum(eval_energy[cls][b]["delta_E"].numel() for cls in ("bdecay", "bg") for b in harness.BUCKETS)
    assert record["tp"] + record["fp"] + record["fn"] + record["tn"] + record["excluded"] == n_eval


def test_run_deploys_the_dedicated_prior_not_the_training_counts(tiny_ds, tiny_cache):
    """Both sides of the CKD comparison must sit at the same, ordering-independent prior."""
    train_energy, eval_energy = harness.extract_split(tiny_ds, ordering="energy")
    counts = harness.dedicated_prior_counts(tiny_ds)
    # A prior swung hard toward background raises every bucket's decision threshold,
    # so the ablation cannot be ignoring the counts it was handed.
    skewed = {b: {"bdecay": 1, "bg": 10_000} for b in harness.BUCKETS}

    baseline = no_ckd_order.run(train_energy, eval_energy, harness.Trainer().config(), counts)
    shifted = no_ckd_order.run(train_energy, eval_energy, harness.Trainer().config(), skewed)
    assert shifted["tp"] < baseline["tp"]


def test_run_writes_plot_set_when_asked(tiny_ds, tiny_cache, tmp_path):
    train_energy, eval_energy = harness.extract_split(tiny_ds, ordering="energy")
    out = tmp_path / "no_ckd"
    counts = harness.dedicated_prior_counts(tiny_ds)
    no_ckd_order.run(train_energy, eval_energy, harness.Trainer().config(), counts, plots_dir=out)
    assert (out / "confusion_matrix_eval.png").is_file()
    assert (out / "roc_eval.png").is_file()
