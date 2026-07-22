import harness

# Resolution-sigma windows swept for the label. 5 is the deployed reference value and
# is included so the curve passes through (and reproduces) the reference at its anchor.
N_STD_VALUES = tuple(range(1, 11))


def run(
    ds: harness.DatasetPaths,
    n_std_values: tuple[int, ...] = N_STD_VALUES,
    seed: int = harness.DEFAULT_SPLIT_SEED,
) -> dict[int, dict]:
    """Sweep the ground-truth label window for one dataset, refitting at each value.

    Streams the ``.sim`` once into a label-free pool, then for every ``n_std`` relabels
    that pool in memory (:func:`pool_to_features`), splits, refits the reference config,
    and evaluates. Only the label window varies — the feature deadzone
    (``ground_truths.calculate_tolerance()``, default ``n_std=5``) is held at its
    deployed 5σ value, so this isolates the label and leaves the model fixed.

    Each fit is decided at the *deployed* operating point: the class priors are the
    dedicated ``<name>_P*.sim`` counts relabeled at the same ``n_std``
    (:func:`~pipeline.priors.pool_prior_counts` over the pool
    :func:`harness.extract_prior_pool` streams once), not the training split's counts —
    so this sweep and the ``dedicated_prior`` deployment sit at the same operating point
    and coincide at ``n_std=5``.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file``/``tra_file`` keys
            plus ``prior_sims`` (the dedicated prior ``.sim`` paths).
        n_std_values: Resolution-sigma windows to sweep (default 1..10).
        seed: Split seed, varied by the multi-seed uncertainty loop (both pool
            extractions are cached, so extra seeds only re-split and refit).

    Returns:
        ``{n_std: metric_record}`` — the eval-split record (counts,
        precision/recall/F1, best_f1, AUC) at each label window.
    """
    # Imported here, after `import harness` above has run its sys.path shim, so the
    # top-level pipeline package `pipeline` resolves regardless of import ordering.
    from pipeline.datasets import pool_to_features, split_features

    pool = harness.extract_pool(ds, ordering="ckd")
    prior_pool = harness.extract_prior_pool(ds)
    results: dict[int, dict] = {}
    for n_std in n_std_values:
        train, eval_data = split_features(pool_to_features(pool, n_std), seed=seed)
        # The prior is relabeled at the same window as the features, so the sweep never
        # mixes a 5σ prior with an n_std label; both pools are streamed once.
        trainer, _ = harness.reference(train, harness.prior_pool_counts(prior_pool, n_std))
        results[n_std] = harness.metric_record(trainer, eval_data)
    return results


def run_heuristic(ds: harness.DatasetPaths, n_std_values: tuple[int, ...] = N_STD_VALUES) -> dict[int, dict]:
    """The energy-only 511 keV photopeak baseline across the same label windows.

    A fit-free, split-free reference for the sweep: the same relabeling ``n_std`` also
    sets the 511 keV window of a plain total-deposited-energy cut, so the baseline
    moves with the label. Streams the ``.sim`` once for all windows and is
    seed-independent, so the driver computes it once per dataset (never per seed).

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file`` keys.
        n_std_values: Windows to sweep (default the module's :data:`N_STD_VALUES`,
            matching the model sweep).

    Returns:
        ``{n_std: record}`` with the heuristic's ``f1_score``/``precision``/
        ``recall``/``fpr`` (plus its raw counts). No ``auc`` (a hard cut has no
        score to rank) and no seed spread (it is deterministic).
    """
    # The leading-digit module name is importable only via importlib, not `import`.
    import importlib

    heuristic = importlib.import_module("511_heuristic")
    return heuristic.sweep(ds, n_std_values)
