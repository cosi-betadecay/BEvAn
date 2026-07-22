"""Ablation A2 — factor contributions on the common >=3-hit population.

What it is:
    A per-dataset measure of how much each physics factor (delta_E, ARM, annihilation
    angle) separates signal from background, with all three scored on the *same*
    events so they are directly comparable. The common population is the bucket-3
    (>=3-hit) events: those are exactly the events where the back-to-back hypothesis
    exists, so delta_E, ARM, and anni are all defined simultaneously. For each factor
    a 1D density (signal vs background) is fit on the training events and the eval
    events are scored by its log-likelihood ratio; the factor's standalone AUC and
    best-F1 are read off that score.

Why this design:
    The earlier version scored each factor on its own pipeline term, but those terms
    live in different buckets (delta_E in bucket 1, the ARM and anni joints in buckets
    2-3) — different populations with different prevalence — so the per-factor F1s
    were not comparable and the delta_E joint double-counted energy. Restricting to
    bucket 3 and using a clean 1D density per factor puts all three on one basis.

What it tries to prove:
    Where the discriminative power lives across the three factors, on equal footing.
    The full-model performance on the same >=3-hit population is drawn as a reference
    line, so each factor's bar reads as "how much of the model's separating power this
    one factor recovers on its own".

Why it is important:
    This is the quantitative core of the interpretability claim — the model's evidence
    decomposed into named, physically-meaningful factors. A black box would need
    post-hoc attribution tooling to approximate the same per-factor picture.

Produces:
    Per dataset, a two-panel plot (F1 left, AUC right) with one bar per factor
    (delta_E, ARM, anni) and a dashed full-model reference, all on the >=3-hit
    population. Run across all datasets so the cross-geometry variation is visible.
"""

from harness import FACTORS, best_f1_threshold, factor_1d_llr, roc_auc


def run(train: dict, eval_data: dict, bucket: int = 3) -> dict[str, dict[str, float]]:
    """Per-factor standalone F1 and AUC on the common >=3-hit population.

    For each factor a 1D density is fit on the training bucket and scored on the eval
    bucket, all over the same events (finite in every factor), so the three factors
    are directly comparable.

    Args:
        train: Nested per-class, per-bucket training feature dict.
        eval_data: The held-out evaluation split.
        bucket: Hit-multiplicity bucket holding the common population (default 3).

    Returns:
        ``{factor: {"f1": ..., "auc": ...}}`` over the factors that score.
    """
    out = {}
    for factor in FACTORS:
        scored = factor_1d_llr(train, eval_data, factor, bucket)
        if scored is None:
            continue
        llr, labels = scored
        out[factor] = {"auc": roc_auc(llr, labels), "f1": best_f1_threshold(llr, labels)[0]}
    return out
