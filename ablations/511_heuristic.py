import harness
import ROOT as M

from modeling.metrics import metrics
from physics.ground_truths import bdecay_label_score, calculate_tolerance
from utils.megalib_types import MSimEvent
from utils.reader_extraction import load_geometry, open_sim_reader


def event_scores(event: MSimEvent) -> tuple[float, float] | None:
    """One event's window-free label and prediction distances (both keV).

    Neither depends on ``n_std``: thresholding each against
    ``calculate_tolerance(n_std)`` recovers the label and the prediction, so a whole
    window sweep costs one ``.sim`` pass (the same trick :func:`bdecay_label_score`
    enables for the model's label window).

    Args:
        event: A MEGAlib simulated event (``MSimEvent``).

    Returns:
        ``(label_score, prediction_distance)`` — ``min|ANNI sum − 511|`` over the
        event's ANNI processes (``inf`` if none), and ``|total deposited − 511|``.
        ``None`` for a hitless event: nothing was deposited, so it is unscorable and
        thrown away (not classified), mirroring the pipeline's routing of events
        with no usable subset.
    """
    n_hits = event.GetNHTs()
    if n_hits == 0:
        return None
    label_score = bdecay_label_score(event)
    total_energy = sum(event.GetHTAt(i).GetEnergy() for i in range(n_hits))
    return label_score, abs(total_energy - 511)


def event_counts(event: MSimEvent, n_std: int) -> tuple[bool, bool] | None:
    """One event's (label, prediction) for the energy-only 511 keV photopeak cut.

    Args:
        event: A MEGAlib simulated event (``MSimEvent``).
        n_std: Resolution-sigma label window; sets the 511 keV tolerance on both
            the truth label and the energy prediction.

    Returns:
        ``(label, prediction)`` — the MC-truth β⁺ tag and whether the event's total
        deposited energy lands in the 511 keV window (strict ``<``, matching
        :func:`ground_truths.ground_truth_bdecay`) — or ``None`` for a hitless event
        (thrown away, not classified).
    """
    scores = event_scores(event)
    if scores is None:
        return None
    tolerance = calculate_tolerance(n_std)
    label_score, prediction_distance = scores
    return label_score < tolerance, prediction_distance < tolerance


def sweep(ds: harness.DatasetPaths, n_std_values: tuple[int, ...]) -> dict[int, dict]:
    """Score the energy-only 511 keV baseline across many label windows in one pass.

    Streams the whole ``.sim`` once (the baseline is fit-free and split-free, so a
    single stream serves every window and every seed), scoring each event once with
    :func:`event_scores` and thresholding it at each ``n_std``. Hitless events are
    unscorable and thrown away (not counted in any cell). Seed-independent, so the
    driver computes it once per dataset.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file`` keys.
        n_std_values: Resolution-sigma windows to sweep (shared with the
            label-window ablation).

    Returns:
        ``{n_std: record}`` where each record is ``{tp, fp, fn, tn}`` plus
        ``precision``/``recall``/``fpr``/``f1_score``.
    """
    tolerances = {n: calculate_tolerance(n) for n in n_std_values}
    counts = {n: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for n in n_std_values}
    reader = open_sim_reader(load_geometry(ds["geo_file"]), ds["sim_file"])
    while True:
        event = reader.GetNextEvent()
        if not event:
            break
        M.SetOwnership(event, True)
        scores = event_scores(event)
        if scores is None:  # hitless event: unscorable, thrown away
            continue
        label_score, prediction_distance = scores
        for n_std, tolerance in tolerances.items():
            label = label_score < tolerance
            prediction = prediction_distance < tolerance
            c = counts[n_std]
            if label and prediction:
                c["tp"] += 1
            elif label:
                c["fn"] += 1
            elif prediction:
                c["fp"] += 1
            else:
                c["tn"] += 1
    reader.Close()

    return {n: {**c, **metrics(c)} for n, c in counts.items()}


def run(ds: harness.DatasetPaths, n_std: int = 5) -> dict:
    """Score the energy-only 511 keV baseline at a single label window.

    A one-window convenience over :func:`sweep`.

    Args:
        ds: Dataset paths with ``name``/``geo_file``/``sim_file`` keys.
        n_std: Resolution-sigma label window (default 5, the deployed reference
            value); shared with the label-window ablation.

    Returns:
        ``{tp, fp, fn, tn}`` plus ``precision``/``recall``/``fpr``/``f1_score``.
    """
    return sweep(ds, (n_std,))[n_std]


def main() -> None:
    """Run the baseline on every discovered dataset and print a one-row-per-dataset table."""
    datasets = harness.discover_datasets()
    header = f"{'dataset':<20} {'tp':>7} {'fp':>7} {'fn':>7} {'tn':>8}  {'f1':>7} {'prec':>7} {'recall':>7} {'fpr':>7}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        r = run(ds)
        print(
            f"{ds['name']:<20} {r['tp']:>7} {r['fp']:>7} {r['fn']:>7} {r['tn']:>8}  "
            f"{r['f1_score']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['fpr']:>7.4f}"
        )


if __name__ == "__main__":
    main()
