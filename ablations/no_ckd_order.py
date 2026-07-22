from pathlib import Path

from harness import Trainer, deploy_prior, metric_record, save_model_plots


def run(
    train_energy: dict,
    eval_energy: dict,
    config: dict,
    prior_counts: dict,
    plots_dir: Path | None = None,
) -> dict:
    """Fit the reference config on energy-ordered features; evaluate.

    The features must have been extracted with ``ordering="energy"`` (the driver
    re-extracts them). The model config is held fixed at the reference's, so only the
    subset ordering differs — isolating the value of the CKD ordering.

    This ablation is decided at the deployed count-prior rule, so it takes the
    dedicated-prior counts rather than the energy-ordered training split's. Those
    counts bucket by raw hit multiplicity, which the subset ordering cannot change,
    so both sides of the comparison sit at the *same* prior — otherwise the delta
    would mix the ordering with the shift in each split's feature-availability
    bucketing.

    Args:
        train_energy: Training features extracted with energy ordering.
        eval_energy: Eval features extracted with energy ordering.
        config: The reference ``Trainer`` config (model held fixed).
        prior_counts: Dedicated-prior counts from
            :func:`harness.dedicated_prior_counts`, shared with the reference.
        plots_dir: When set, save this variant's diagnostic plots (density terms,
            eval confusion matrix, ROC/PR) there — the density heatmaps show how
            energy ordering reshapes each term versus the reference's CKD ones.

    Returns:
        The eval-split metric record under energy-ordered features.
    """
    trainer = deploy_prior(Trainer(**config).fit(train_energy), prior_counts)
    if plots_dir is not None:
        save_model_plots(trainer, eval_energy, plots_dir)
    return metric_record(trainer, eval_energy)
