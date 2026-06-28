from harness import Trainer, metric_record


def run(train: dict, eval_data: dict, config: dict) -> dict:
    """Fit the champion config but skip calibration; evaluate at the count-prior point.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split, same nested structure.
        config: The champion ``Trainer`` config (reused so only calibration differs).

    Returns:
        The eval-split metric record (counts, precision/recall/F1, best_f1, AUC) at
        the uncalibrated count-prior operating point.
    """
    trainer = Trainer(**config).fit(train)
    return metric_record(trainer, eval_data)
