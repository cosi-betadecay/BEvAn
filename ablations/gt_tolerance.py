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
    A plot of F1, AUC, precision, and recall versus ``n_std`` (1..10 by default), averaged
    across the Ge geometries — roughly flat curves mean the conclusions are robust to the
    window. Per-dataset detail goes to CSV. One re-extraction per ``n_std`` value (the
    labels change), so it is the most expensive ablation to run. ``n_std = 3`` reuses the
    champion's own feature cache, so that anchor point reproduces the champion exactly.
"""

import harness

# Resolution-sigma windows swept for the label. 3 is the deployed champion value and
# is included so the curve passes through (and reproduces) the champion at its anchor.
N_STD_VALUES = tuple(range(1, 11))


def run(
    ds: dict[str, str],
    n_std_values: tuple[int, ...] = N_STD_VALUES,
    n_iter: int = 50,
    calibrate: bool = True,
) -> dict[int, dict]:
    """Sweep the ground-truth label window for one dataset, refitting at each value.

    For every ``n_std`` the label is recomputed, which forces a fresh feature
    extraction (signal/background membership changes, so the cache cannot be reused
    except at the champion anchor ``n_std = 3``), then the champion config is refit
    and evaluated. Only the label window varies — the feature deadzone
    (``physics_factors.E_TOLERANCE``) is held at its deployed 3σ value, so this
    isolates the label and leaves the model fixed.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        n_std_values: Resolution-sigma windows to sweep (default 1..10).
        n_iter: Champion-seeded search candidates per fit (forwarded to the champion).
        calibrate: Whether the refit champion calibrates its threshold offset.

    Returns:
        ``{n_std: metric_record}`` — the eval-split record (counts,
        precision/recall/F1, best_f1, AUC) at each label window.
    """
    results: dict[int, dict] = {}
    for n_std in n_std_values:
        train, eval_data = harness.extract_split(ds, ordering="ckd", gt_n_std=n_std)
        trainer, _ = harness.champion(train, n_iter=n_iter, calibrate=calibrate)
        results[n_std] = harness.metric_record(trainer, eval_data)
    return results
