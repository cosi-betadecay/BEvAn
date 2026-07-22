"""The energy-only 511 keV photopeak baseline (ablations/511_heuristic.py).

The leading-digit module name is importable only via importlib, not ``import``.
"""

import importlib

import label_window

heuristic = importlib.import_module("511_heuristic")


def _counts_via_event_counts(ds, n_std: int) -> dict[str, int]:
    """Reference confusion counts, streamed one event at a time through event_counts."""
    import ROOT as M

    from utils.reader_extraction import load_geometry, open_sim_reader

    reader = open_sim_reader(load_geometry(ds["geo_file"]), ds["sim_file"])
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    while True:
        event = reader.GetNextEvent()
        if not event:
            break
        M.SetOwnership(event, True)
        result = heuristic.event_counts(event, n_std)
        if result is None:  # hitless event: thrown away, not classified
            continue
        label, prediction = result
        key = "tp" if (label and prediction) else "fn" if label else "fp" if prediction else "tn"
        counts[key] += 1
    reader.Close()
    return counts


def test_sweep_has_the_four_metrics_and_no_auc(tiny_ds):
    results = heuristic.sweep(tiny_ds, (1, 5))
    assert set(results) == {1, 5}
    for record in results.values():
        assert {"tp", "fp", "fn", "tn", "f1_score", "precision", "recall", "fpr"} <= set(record)
        assert "auc" not in record  # a hard cut has no ranked score


def test_run_is_the_single_window_of_sweep(tiny_ds):
    assert heuristic.run(tiny_ds, n_std=5) == heuristic.sweep(tiny_ds, (5,))[5]


def test_single_pass_sweep_matches_per_event_counts(tiny_ds):
    # The one-pass thresholding must reproduce the per-event event_counts path exactly.
    results = heuristic.sweep(tiny_ds, (1, 5))
    for n_std in (1, 5):
        expected = _counts_via_event_counts(tiny_ds, n_std)
        assert {k: results[n_std][k] for k in ("tp", "fp", "fn", "tn")} == expected


def test_every_scorable_event_is_counted_once_per_window(tiny_ds):
    results = heuristic.sweep(tiny_ds, (1, 5))
    totals = {n: sum(results[n][k] for k in ("tp", "fp", "fn", "tn")) for n in results}
    assert totals[1] == totals[5]  # same event stream, every scorable event classified


def test_hitless_events_are_thrown_away(tiny_ds, synthetic_dataset):
    # The fixture ends in a zero-hit event; it must not land in any cell, so the
    # counted total equals the number of non-empty events, not the full stream.
    results = heuristic.sweep(tiny_ds, (5,))
    total = sum(results[5][k] for k in ("tp", "fp", "fn", "tn"))
    assert total == synthetic_dataset["n_nonempty"]


def test_wider_window_labels_at_least_as_many_positives(tiny_ds):
    results = heuristic.sweep(tiny_ds, (1, 5))
    positives = {n: results[n]["tp"] + results[n]["fn"] for n in results}
    assert positives[1] <= positives[5]  # tolerance grows with n_std


def test_sweep_is_deterministic(tiny_ds):
    assert heuristic.sweep(tiny_ds, (1, 5)) == heuristic.sweep(tiny_ds, (1, 5))


def test_run_heuristic_covers_the_default_windows(tiny_ds):
    results = label_window.run_heuristic(tiny_ds)
    assert set(results) == set(label_window.N_STD_VALUES)
