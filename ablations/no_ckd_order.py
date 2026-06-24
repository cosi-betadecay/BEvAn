"""Ablation A6 — no CKD ordering (energy-order hit subsets instead).

What it is:
    Event processing currently enumerates hit subsets (up to size 7, Boggs & Jean
    2000) and orders each one by Compton-kinematic-discrepancy consistency — the
    mean squared residual between the geometric and kinematic scatter angles —
    keeping only physically consistent orderings before the features are computed.
    This row replaces that with a plain energy ordering (E1 >= E2 >= ...), the
    cheap heuristic, and recomputes the features from the re-ordered subsets.

What it tries to prove:
    Whether the CKD ordering — a core piece of the physics pipeline — actually
    improves discrimination, or whether a trivial energy sort would have done as
    well. Because the ordering feeds ARM and the back-to-back score (it decides
    which scatter sequence each feature sees), this isolates the value of the
    kinematic-consistency step independent of the features themselves.

Why it is important:
    The CKD ordering is one of the method's non-obvious design choices and a
    reviewer will ask whether it is doing real work or is ceremony. This row
    justifies (or retires) it. NOTE: this is a feature-level ablation — it changes
    what event_processing produces, so it requires re-extraction and cannot run
    off cached features.
"""
