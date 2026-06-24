"""Ablation A1 — champion baseline (the control row).

What it is:
    The unmodified champion model exactly as shipped: per-bucket Bayesian
    likelihood over the three hit-multiplicity buckets (1: delta_E; 2: delta_E +
    ARM; 3: delta_E + ARM + anni), CKD-ordered hit subsets, F1 threshold
    calibration on, equal term weights. No switch is flipped — this is the
    reference every other ablation row is measured against.

What it tries to prove:
    That the harness reproduces the established champion result (eval F1 0.8935,
    commit 86bb140) byte-for-byte on the fixed split. If ``base`` does not land
    on the champion number, no downstream ablation can be trusted, because the
    whole point of the leave-one-out design is that only the named component
    changes between rows.

Why it is important:
    Every claim in the ablation section is differential ("removing X costs Y F1").
    A differential is only meaningful against an identical, verified control, so
    this row is the anchor that makes the rest of the table comparable — and the
    guard that catches any accidental drift in the shared load -> fit -> eval path.
"""
