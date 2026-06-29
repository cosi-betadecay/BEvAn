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

    Streams the ``.sim`` once into a label-free pool, then for every ``n_std`` relabels
    that pool in memory (:func:`pool_to_features`), splits, refits the champion config,
    and evaluates. Only the label window varies — the feature deadzone
    (``physics_factors.calculate_tolerance()``, default ``n_std=3``) is held at its
    deployed 3σ value, so this isolates the label and leaves the model fixed.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys.
        n_std_values: Resolution-sigma windows to sweep (default 1..10).
        n_iter: Champion-seeded search candidates per fit (forwarded to the champion).
        calibrate: Whether the refit champion calibrates its threshold offset.

    Returns:
        ``{n_std: metric_record}`` — the eval-split record (counts,
        precision/recall/F1, best_f1, AUC) at each label window.
    """
    # Imported here, after `import harness` above has run its sys.path shim, so the
    # top-level pipeline package `pipeline` resolves regardless of import ordering.
    from pipeline.datasets import pool_to_features, split_features

    pool = harness.extract_pool(ds, ordering="ckd")
    results: dict[int, dict] = {}
    for n_std in n_std_values:
        train, eval_data = split_features(pool_to_features(pool, n_std))
        trainer, _ = harness.champion(train, n_iter=n_iter, calibrate=calibrate)
        results[n_std] = harness.metric_record(trainer, eval_data)
    return results
