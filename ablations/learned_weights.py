import math
from pathlib import Path

from harness import (
    FACTORS,
    Evaluator,
    Trainer,
    best_f1_threshold,
    fit_logistic,
    metrics,
    roc_auc,
    save_density_terms,
    save_score_plots,
    term_llr_columns,
)


def run(train: dict, eval_data: dict, config: dict, plots_dir: Path | None = None) -> dict:
    """Fit logistic weights on the per-term LLRs and evaluate with them.

    Builds the reference model, fits a logistic regression over the three per-term
    log-likelihood ratios on the training split (the MLE weighting), calibrates the
    decision threshold on the training logits (F1-optimal), then scores the eval
    split at that threshold. The fitted weights are returned alongside the
    metrics — they are themselves the result (expected near equal).

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.
        config: The reference ``Trainer`` config.
        plots_dir: When set, save this variant's diagnostic plots there. The
            density terms are the underlying trainer's (the weights reweight the
            terms, they do not refit them); the confusion matrix and ROC/PR use
            the logistic logits, this ablation's actual decision score.

    Returns:
        The eval-split metric record plus the fitted per-factor weights, one flat
        ``w_<factor>`` scalar each, and the ``bias``. Flat rather than a nested dict
        so :func:`harness.aggregate_seed_records` averages them across seeds like
        every other column instead of carrying the first seed's alone, which is what
        lets :func:`tables.save_comparison_csv` persist them beside the metrics.
    """
    trainer = Trainer(**config).fit(train)
    evaluator = Evaluator(trainer)

    x_train, y_train = term_llr_columns(evaluator, train)
    weights, bias = fit_logistic(x_train, y_train)

    # Calibrate the decision threshold on the training logits (F1-optimal cut).
    # Without this the F1 comparison would conflate the weighting with an
    # arbitrary logit>=0 boundary; AUC/best_f1 (threshold-free) are unaffected
    # either way.
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
    if plots_dir is not None:
        save_density_terms(trainer, plots_dir)
        save_score_plots(logits, ground_truths, counts, plots_dir)
    return {
        **counts,
        **metrics(counts),
        "best_f1": best_f1_threshold(logits, ground_truths)[0],
        "auc": roc_auc(logits, ground_truths),
        **{f"w_{factor}": float(w) for factor, w in zip(FACTORS, weights, strict=True)},
        "bias": bias,
    }
