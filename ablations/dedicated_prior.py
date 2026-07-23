from pathlib import Path

import harness


def run(
    ds: harness.DatasetPaths,
    train: dict,
    eval_data: dict,
    plots_dir: Path | None,
    prior_counts: dict[int, dict[str, int]] | None = None,
) -> dict:
    """Deploy the reference at the dedicated-prior operating point for one dataset.

    The former ``batch_analysis.py`` run, folded into the ablation driver: the class
    priors are counted from the dataset's dedicated ``<name>_P*.sim`` files
    (:mod:`pipeline.priors`) and set on a freshly fitted reference, so the reported
    records sit at the actual deployed operating point. A fresh trainer is fitted
    (cheap next to the cached extraction) so this entry stays runnable on its own —
    ``--ablations dedicated_prior`` — without a driver-built reference. Under the full
    driver the shared reference now carries the same prior, so the two coincide.

    Args:
        ds: Dataset paths with ``name``/``geo_file`` keys plus ``prior_sims``
            (the dedicated prior ``.sim`` paths from :func:`harness.discover_datasets`).
        train: Reference (CKD) training features.
        eval_data: Reference (CKD) eval features.
        plots_dir: Directory for the deployed model's diagnostic plots (density
            terms, eval confusion matrix, ROC/PR, and the tagged-cone sky figure),
            or None to skip all figures (the extra seeds of a multi-seed run).
        prior_counts: Already-counted per-bucket prior counts to reuse (the driver
            passes :func:`harness.dedicated_prior_counts`, shared with every other
            prior-dependent ablation); None counts them from ``prior_sims`` with a
            fresh stream, for standalone use.

    Returns:
        ``{"prior_counts", "priors", "train", "eval"}`` — the counted priors and
        the deployed-point metric record per split.
    """
    # Imported here, after `import harness` above has run its sys.path shim, so the
    # top-level pipeline package `pipeline` resolves regardless of import ordering.
    from pipeline.priors import apply_prior_counts, bucket_priors, count_prior_events, validate_prior_counts

    if prior_counts is None:
        prior_counts = count_prior_events(ds["geo_file"], ds["prior_sims"])
    validate_prior_counts(prior_counts)
    priors = bucket_priors(prior_counts)

    trainer, _ = harness.reference(train)
    apply_prior_counts(trainer, prior_counts)
    if plots_dir is not None:
        harness.save_model_plots(trainer, eval_data, plots_dir)
        save_sky_figure(ds, plots_dir)
    return {
        "prior_counts": prior_counts,
        "priors": priors,
        "train": harness.metric_record(trainer, train),
        "eval": harness.metric_record(trainer, eval_data),
    }


def save_sky_figure(ds: harness.DatasetPaths, plots_dir: Path) -> None:
    """Save (and W&B-log) the all-events Compton-cone sky image.

    Draws the all-events backprojection cached at extraction — the same figure
    ``analysis.py`` produces per run.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``tra_file`` keys.
        plots_dir: Directory for this dataset's dedicated_prior figures.
    """
    from utils.local_results import save_sky_backprojection
    from utils.wandb_logging import log_sky_backprojection

    _, _, sky_all, n_all = harness.extract_keyed(ds)
    imager = harness.sky_imager(ds)
    save_sky_backprojection(sky_all, n_all, plots_dir, imager.phi_range, imager.theta_range)
    log_sky_backprojection(
        sky_all, n_all, imager.phi_range, imager.theta_range, split_name=f"{ds['name']}/dedicated_prior"
    )
