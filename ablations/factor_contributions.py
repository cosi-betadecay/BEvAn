"""Ablation A2 — factor contributions (how much each physics factor drives the decision).

What it is:
    A contribution analysis over the three physics factors (delta_E, ARM,
    annihilation angle) rather than a single stripped-down model. It exploits the
    fact that the classifier's score is additive in log-space —
    ``log R = LLR_delta_E + LLR_ARM + LLR_anni`` — so a decision can be decomposed
    exactly into per-factor pieces with no post-hoc approximation. It reports two
    complementary views:

      1. Per-event evidence decomposition: how many nats of log-likelihood ratio
         each factor contributes toward signal vs background, averaged over events
         (and split by correct/incorrect, by bucket, by class) — the model reading
         off its own reasoning.
      2. Per-factor performance attribution: each factor's standalone discriminative
         power (single-term F1/AUC), its marginal/leave-one-out contribution (F1 cost
         of removing it), and its Shapley value over the {delta_E, ARM, anni} set —
         the fair allocation that splits shared credit correctly.

    The Shapley step is the load-bearing one: ARM's theta_kin is computed from the
    deposited energies, so delta_E and ARM share information. Naively summing
    standalone contributions double-counts that overlap; the Shapley decomposition
    (and the standalone-vs-marginal gap it exposes) is what makes the attribution
    honest about the redundancy.

What it tries to prove:
    Where the discriminative information actually lives across the three factors,
    in one coherent, redundancy-aware analysis. It absorbs the old energy-only
    result as a special case — delta_E's standalone / Shapley contribution *is* the
    "energy window alone" number — so it still answers the circularity attack (label
    and headline feature are both 511 keV energy) while additionally quantifying ARM
    and anni in the same framework, and showing how much of ARM's apparent value is
    really the delta_E double-count.

Why it is important:
    This is the quantitative core of the interpretability claim. The whole
    "interpretable vs black-box" argument is that the model decomposes its own
    decisions exactly, where a black box needs SHAP/attribution tooling to
    approximate the same thing; this file is that decomposition, computed from the
    model's structure. For the method/instrument claim it justifies each term with
    a fair contribution rather than a misleading single-feature score, and the
    expected finding — energy dominates, geometry adds little once its overlap with
    energy is removed — is delivered as a principled attribution, not an assertion.

    NOTE: arm_as_deltaE.py is kept as a separate ablation; it is a physics-specific
    separation of ARM's geometry from the energy it double-counts that the generic
    Shapley-over-terms decomposition here cannot express.
"""
