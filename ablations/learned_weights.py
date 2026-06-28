import math

from harness import (
    FACTORS,
    Evaluator,
    Trainer,
    best_f1_threshold,
    fit_logistic,
    metrics,
    roc_auc,
    term_llr_columns,
)


def run(train: dict, eval_data: dict, config: dict) -> dict:
    """Fit logistic weights on the per-term LLRs and evaluate with them.

    Builds the champion model, fits a logistic regression over the three per-term
    log-likelihood ratios on the training split (the MLE weighting), calibrates the
    decision threshold on the training logits (F1-optimal, same spirit as our model's
    calibration), then scores the eval split at that threshold. The fitted weights are
    returned alongside the metrics — they are themselves the result (expected near equal).

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.
        config: The champion ``Trainer`` config.

    Returns:
        The eval-split metric record plus the fitted ``weights`` and ``bias``.
    """
    trainer = Trainer(**config).fit(train)
    evaluator = Evaluator(trainer)

    x_train, y_train = term_llr_columns(evaluator, train)
    weights, bias = fit_logistic(x_train, y_train)

    # Calibrate the decision threshold on the training logits (F1-optimal cut),
    # mirroring how our model calibrates its operating point. Without this the F1
    # comparison would conflate the weighting with an arbitrary logit>=0 boundary;
    # AUC/best_f1 (threshold-free) are unaffected either way.
    train_logits = (x_train.double() @ weights + bias).float()
    _, threshold = best_f1_threshold(train_logits, y_train.bool())
    if math.isnan(threshold):  # no positives on train -> fall back to the logistic boundary
        threshold = 0.0

    x_eval, y_eval = term_llr_columns(evaluator, eval_data)
    logits = (x_eval.double() @ weights + bias).float()
    ground_truths = y_eval.bool()
    predictions = logits >= threshold
    counts = {
        "tp": int((predictions & ground_truths).sum()),
        "fp": int((predictions & ~ground_truths).sum()),
        "fn": int((~predictions & ground_truths).sum()),
        "tn": int((~predictions & ~ground_truths).sum()),
        "excluded": 0,
    }
    return {
        **counts,
        **metrics(counts),
        "best_f1": best_f1_threshold(logits, ground_truths)[0],
        "auc": roc_auc(logits, ground_truths),
        "weights": {factor: float(w) for factor, w in zip(FACTORS, weights, strict=True)},
        "bias": bias,
    }
