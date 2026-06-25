"""Robustness — ground-truth tolerance sweep (how much the label window matters).

What it is:
    A sweep of the ground truth's energy-window tolerance rather than a model ablation.
    ``ground_truth_bdecay`` labels an event β⁺ when an ANNI secondary's hits sum to within
    ``n_std`` sigma of 511 keV (the deployed label uses ``n_std = 3``). This sweep re-runs
    the whole pipeline at several ``n_std`` values: each value **re-labels** the events,
    so it forces a fresh feature extraction (the label decides which events are signal vs
    background — it cannot reuse the cache), then refits the champion and evaluates.
    ``n_std = 3`` reproduces the champion exactly.

    NOTE: this changes the *task*, not just a knob — a tighter window keeps fewer, cleaner
    signal events and a looser one admits more, so the metrics are expected to move. The
    point is not that the numbers are identical but that the *picture* is stable.

What it tries to prove:
    That the study's conclusions do not hinge on the one hand-set 3σ photopeak window —
    that energy still dominates and the model's performance stays in the same band whether
    the window is tightened or loosened. It pre-empts the sharpest attack on the label:
    "your result is an artifact of an arbitrary tolerance."

Why it is important:
    The ground truth is a photopeak criterion, so the *entire* classification task is
    defined by this tolerance. That makes it the most attackable foundation in the paper,
    and a robustness sweep is the cleanest defense — it shows the findings are a property
    of the physics, not of a particular cut value.

Produces:
    A plot of F1, AUC, precision, and recall versus ``n_std`` (e.g. {1, 2, 3}), averaged
    across the Ge geometries — roughly flat curves mean the conclusions are robust to the
    window. Per-dataset detail goes to CSV. One re-extraction per ``n_std`` value (the
    labels change), so it is the most expensive ablation to run.
"""
