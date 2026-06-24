"""Ablation A9 — naive 511 keV cut (the bare energy-window baseline).

What it is:
    The simplest possible classifier: predict β⁺ iff the event has a subset within a
    window of 511 keV, i.e. ``delta_E <= c``. No densities, no geometry, no Bayesian
    model — just a threshold on the energy distance from the annihilation line. The
    window ``c`` is calibrated on the training split (F1-optimal) and applied to eval,
    over the *same events the full model scores* (apples-to-apples via
    :func:`harness.pooled_feature`).

What it tries to prove:
    How much the full physics pipeline actually buys over a bare energy cut. The
    factor-contribution analysis already implies ΔE carries almost all the separating
    power; this turns that into the head-to-head number a reviewer's first question
    demands — "isn't this just a 511 keV window?" — answered explicitly rather than
    by inference.

Why it is important:
    The ground truth is itself a photopeak criterion and ΔE is the energy distance to
    511 keV, so the bare cut is the natural null model. The gap between it and the full
    model is exactly what the interpretable machinery is worth — and owning that number
    honestly (even if small) is what lets the paper pair "interpretable" with "and it
    costs essentially nothing over a cut, while explaining itself".

Produces:
    A plot of F1, AUC, precision, and recall for the naive cut vs the full model across
    all datasets, plus the deployed window half-width (``window_keV``) in the record.
"""

from harness import best_f1_threshold, metrics, pooled_feature, roc_auc


def run(train: dict, eval_data: dict) -> dict:
    """Calibrate a 511 keV energy window on train and score eval, on the model's population.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.

    Returns:
        The eval-split metric record (counts, precision/recall/F1, best_f1, AUC) plus
        ``window_keV``, the deployed ``|ΔE| <= c`` half-width.
    """
    dE_train, y_train = pooled_feature(train, "delta_E")
    dE_eval, y_eval = pooled_feature(eval_data, "delta_E")
    # Smaller ΔE is more signal-like, so score on -ΔE and predict signal at score >= cut.
    score_train, score_eval = -dE_train, -dE_eval
    _, threshold = best_f1_threshold(score_train, y_train)
    if threshold != threshold:  # no positives on train -> degenerate; keep the window closed
        threshold = 0.0

    predictions = score_eval >= threshold
    counts = {
        "tp": int((predictions & y_eval).sum()),
        "fp": int((predictions & ~y_eval).sum()),
        "fn": int((~predictions & y_eval).sum()),
        "tn": int((~predictions & ~y_eval).sum()),
        "excluded": 0,
    }
    return {
        **counts,
        **metrics(counts),
        "best_f1": best_f1_threshold(score_eval, y_eval)[0],
        "auc": roc_auc(score_eval, y_eval),
        "window_keV": -threshold,
    }
