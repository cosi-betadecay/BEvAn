"""Ablation A7 — no calibration (raw count-prior operating point).

What it is:
    The champion calibrates a global decision-threshold offset for F1 on the
    training split (guarded; a zero offset keeps the count-prior default) and
    deploys it. This row disables that step and classifies at the raw count-prior
    posterior operating point — the bare Bayesian decision rule with no
    threshold tuning.

What it tries to prove:
    The contribution of the calibration step to generalization across datasets.
    This is the model's strongest *positive* methodological result: calibration is
    a robust win (SPILike eval F1 0.84 -> 0.88, balloon 0.074 -> 0.273) that moves
    only the threshold — the underlying densities are byte-identical, so the gain
    is purely a better-placed operating point, not a different model.

Why it is important:
    Most ablations subtract to show a component is marginal; this one subtracts to
    show a component is load-bearing. For a reusable-instrument claim it is the
    cleanest evidence that the pipeline transfers its operating point sensibly to
    new geometries, and it cleanly separates "the model ranks events well" (AUC,
    unchanged) from "the model decides well" (F1, calibration-dependent).
"""
