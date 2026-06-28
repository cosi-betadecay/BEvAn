from harness import Trainer, metric_record


def run(train_energy: dict, eval_energy: dict, config: dict) -> dict:
    """Fit the champion config on energy-ordered features; evaluate.

    The features must have been extracted with ``ordering="energy"`` (the driver
    re-extracts them). The model config is held fixed at the champion's, so only the
    subset ordering differs — isolating the value of the CKD ordering.

    Args:
        train_energy: Training features extracted with energy ordering.
        eval_energy: Eval features extracted with energy ordering.
        config: The champion ``Trainer`` config (model held fixed).

    Returns:
        The eval-split metric record under energy-ordered features.
    """
    trainer = Trainer(**config).fit(train_energy)
    return metric_record(trainer, eval_energy)
