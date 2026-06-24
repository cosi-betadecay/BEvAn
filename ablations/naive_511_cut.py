"""Ablation A9 — straight 511 keV cut (the bare total-energy window baseline).

What it is:
    The dumbest possible classifier, exactly as one would do by hand: take the event's
    total deposited energy (``E_total`` = sum of all hit energies) and predict β⁺ iff it
    lies within a fixed window of 511 keV — by default ±3 keV, i.e. [508, 514] keV,
    matching the ~3σ tolerance the ground truth itself uses. No subsets, no engineered
    delta_E, no densities, no Bayesian model — just an energy window.

    NOTE: this is deliberately *not* a cut on the model's ``delta_E`` feature.
    ``delta_E = min|subset_energy − 511|`` is an engineered quantity built by enumerating
    hit subsets to find the best match to 511 — it is essentially the model's strongest
    feature, so cutting on it just reproduces the model. The straight cut uses the raw
    total event energy, which is genuinely independent of that machinery.

What it tries to prove:
    What the full physics pipeline buys over a hand-drawn energy window — the literal
    answer to "isn't this just a 511 keV cut?". Because a partially-absorbed annihilation
    photon deposits less than 511 keV total (the model's subset-based delta_E still
    catches it, the total-energy window does not), the gap between this baseline and the
    model is exactly the value of the subset reconstruction + likelihood machinery.

Why it is important:
    The ground truth is a photopeak criterion, so a total-energy window is the natural
    null model. Owning the gap honestly is what lets the paper pair "interpretable" with
    "and it earns its place over a bare cut, while explaining itself".

Produces:
    A plot of F1, AUC, precision, and recall for the straight cut vs the full model
    across all datasets; ``window_keV`` records the deployed half-width.
"""

from harness import best_f1_threshold, metrics, pooled_feature, roc_auc

ANNIHILATION_KEV = 511.0
WINDOW_KEV = 3.0  # ±3 keV -> [508, 514], matching the ground-truth ~3σ tolerance


def run(eval_data: dict, window_keV: float = WINDOW_KEV) -> dict:
    """Classify eval events by a fixed total-energy window around 511 keV.

    Evaluated on the same population the full model scores (via
    :func:`harness.pooled_feature`), so the head-to-head is apples-to-apples.

    Args:
        eval_data: The held-out evaluation split.
        window_keV: Half-width of the energy window; ``|E_total - 511| <= window_keV``.

    Returns:
        The eval-split metric record (counts, precision/recall/F1, best_f1, AUC) plus
        ``window_keV``. ``best_f1``/``auc`` use the threshold-free score ``-|E_total-511|``.
    """
    e_total, y_eval = pooled_feature(eval_data, "E_total")
    deviation = (e_total - ANNIHILATION_KEV).abs()
    predictions = deviation <= window_keV  # the [508, 514] keV window
    counts = {
        "tp": int((predictions & y_eval).sum()),
        "fp": int((predictions & ~y_eval).sum()),
        "fn": int((~predictions & y_eval).sum()),
        "tn": int((~predictions & ~y_eval).sum()),
        "excluded": 0,
    }
    score = -deviation  # closer to 511 keV = more signal-like (threshold-free diagnostics)
    return {
        **counts,
        **metrics(counts),
        "best_f1": best_f1_threshold(score, y_eval)[0],
        "auc": roc_auc(score, y_eval),
        "window_keV": window_keV,
    }
