"""Truth-graded harness for the blind LOR annihilation reconstruction.

Iterates every event in Activation.sim once, runs the blind two-photon
reconstruction (``physics.annihilation_lor``) on the measured hits, and
extracts the simulation's ANNI ground truth (photon axis + which photons
actually deposited fully) from the IA lines. Three questions, one test each:

1. Topology census — what fraction of labeled-signal events actually have
   two complete photon chains (the textbook case the LOR needs) vs one vs
   none? This sizes the fallback problem before any feature design bets.
2. Axis accuracy — for events where the LOR is reconstructable, how far is
   the blind axis from the true ANNI emission axis?
3. Class overlap — same protocol as test_overlaps.py, applied to the new
   score: are the signal/background score distributions separable?

ANNI truth is used ONLY for grading; the reconstruction sees hits alone.

The events are read with the pure-Python ``utils.sim_text_reader`` (no
ROOT/MEGAlib build needed): the .sim text file carries the same positions,
energies, and IA records the PyROOT path exposes, and the signal label is
replicated from ``physics.ground_truths.ground_truth_bdecay``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import wandb
from mathematics.calculations import calculate_tolerance
from physics.annihilation_lor import MAX_LOR_HITS, annihilation_lor
from utils.sim_text_reader import SimTextEvent, event_is_bdecay, iter_sim_events

PHOTON_KEV = 511.0
SIM_FILE = "data/Activation.sim"


# ---------------------------------------------------------------------------
# Truth extraction (simulation-only; never feeds the reconstruction)
# ---------------------------------------------------------------------------


def _descendant_sets(event: SimTextEvent, root_ids):
    """For each root IA id, the set of IA ids whose parent chain passes
    through it (including the root itself). IA origins always reference
    earlier IAs, so one ascending pass suffices."""
    sets = {root: {root} for root in root_ids}
    for ia in sorted(event.ias, key=lambda r: r.ia_id):
        for members in sets.values():
            if ia.origin_id in members:
                members.add(ia.ia_id)
    return sets


def _photon_deposit(event: SimTextEvent, descendant_ids) -> float:
    """Energy deposited in hits by any IA in ``descendant_ids`` [keV]."""
    return sum(hit.energy for hit in event.hits if any(o in descendant_ids for o in hit.origins))


def _extract_annihilations(event: SimTextEvent, tolerance: float):
    """All annihilations in the event as dicts with the true axis and the
    per-photon deposited energies.

    Each annihilation at rest emits two ANNI IA lines sharing OriginID and
    position, with exactly negated photon directions. In-flight annihilation
    breaks the negation but still yields two lines per origin.
    """
    anni = [ia for ia in event.ias if ia.process == "ANNI"]
    if not anni:
        return []

    desc = _descendant_sets(event, [ia.ia_id for ia in anni])

    by_origin = {}
    for ia in anni:
        by_origin.setdefault(ia.origin_id, []).append(ia)

    out = []
    for photons in by_origin.values():
        deposits = [_photon_deposit(event, desc[p.ia_id]) for p in photons]
        complete = [abs(dep - PHOTON_KEV) < tolerance for dep in deposits]
        out.append(
            {
                "axis": np.array(photons[0].secondary_dir, dtype=np.float64),
                "photon_energies": [p.secondary_energy for p in photons],
                "deposits": deposits,
                "n_complete": int(sum(complete)),
                "pair_complete": sum(complete) >= 2,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Session fixture: one pass over the full sim file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def lor_harness():
    """Single pass over Activation.sim: blind reconstruction + ANNI truth
    per event. Events with zero hits are skipped, matching the production
    pipeline's population."""
    tolerance = calculate_tolerance()

    labels = []
    n_hits_list = []
    scores = []
    angular_errors = []  # min angle vs true axes [deg], NaN if not gradable
    n_complete_photons = []  # per event: max complete photons of any single annihilation
    has_complete_pair = []  # per event: any annihilation with both photons complete

    for event in iter_sim_events(SIM_FILE):
        n_hits = len(event.hits)
        if n_hits == 0:
            continue

        positions = np.array([hit.position for hit in event.hits], dtype=np.float64)
        energies = np.array([hit.energy for hit in event.hits], dtype=np.float64)

        result = annihilation_lor(positions, energies)
        truth = _extract_annihilations(event, tolerance)

        labels.append(event_is_bdecay(event, PHOTON_KEV, tolerance))
        n_hits_list.append(n_hits)
        scores.append(result.score)

        if truth:
            n_complete_photons.append(max(t["n_complete"] for t in truth))
            has_complete_pair.append(any(t["pair_complete"] for t in truth))
        else:
            n_complete_photons.append(0)
            has_complete_pair.append(False)

        if result.axis is not None and truth:
            errs = []
            for t in truth:
                cosang = abs(float(np.dot(result.axis, t["axis"])))
                errs.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            angular_errors.append(min(errs))
        else:
            angular_errors.append(float("nan"))

    return {
        "labels": np.array(labels, dtype=bool),
        "n_hits": np.array(n_hits_list, dtype=np.int64),
        "scores": np.array(scores, dtype=np.float64),
        "angular_errors": np.array(angular_errors, dtype=np.float64),
        "n_complete_photons": np.array(n_complete_photons, dtype=np.int64),
        "has_complete_pair": np.array(has_complete_pair, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Overlap helpers (same protocol as test_overlaps.py)
# ---------------------------------------------------------------------------


def _density(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype="float64")
    return counts.astype("float64") / total / widths


def _bhattacharyya(p, q, edges):
    widths = np.diff(edges)
    return float(np.sum(np.sqrt(p * q) * widths))


def _total_variation(p, q, edges):
    widths = np.diff(edges)
    return 0.5 * float(np.sum(np.abs(p - q) * widths))


# ---------------------------------------------------------------------------
# 1. Topology census — sizes the fallback problem
# ---------------------------------------------------------------------------


def test_signal_topology_census(lor_harness, wandb_run):
    """Among labeled-signal events, how many have a fully absorbed photon
    PAIR (the LOR's textbook case) vs exactly one complete photon (escape /
    partial deposit — needs the cone fallback) vs none?

    Also cross-validates the transitive-descendant truth extraction against
    the label: ``ground_truth_bdecay`` marks an event signal iff some photon
    deposits ~511 keV, so nearly every labeled event must show >=1 complete
    photon here. A low fraction means the truth extraction is wired wrong,
    not that the physics changed.
    """
    t = lor_harness
    sig = t["labels"]
    n_sig = int(sig.sum())
    assert n_sig > 0, "no labeled signal events found"

    n_complete = t["n_complete_photons"][sig]
    frac_pair = float(t["has_complete_pair"][sig].mean())
    frac_one = float((n_complete == 1).mean())
    frac_at_least_one = float((n_complete >= 1).mean())
    frac_none = float((n_complete == 0).mean())
    frac_too_large = float((t["n_hits"][sig] > MAX_LOR_HITS).mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["pair complete", "exactly one", "none"]
    vals = [frac_pair, frac_one, frac_none]
    bars = ax.bar(cats, vals, color=["tab:green", "tab:orange", "tab:red"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("fraction of signal events")
    ax.set_title(f"Signal topology census (n={n_sig})")

    wandb_run.log(
        {
            "lor/census/n_signal": n_sig,
            "lor/census/frac_pair_complete": frac_pair,
            "lor/census/frac_exactly_one_complete": frac_one,
            "lor/census/frac_at_least_one_complete": frac_at_least_one,
            "lor/census/frac_none_complete": frac_none,
            "lor/census/frac_signal_above_hit_cap": frac_too_large,
            "lor/census/bars": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert frac_at_least_one >= 0.95, (
        f"only {frac_at_least_one:.3f} of labeled-signal events show a complete "
        "photon in the truth extraction — the label requires one, so the "
        "descendant/deposit accounting disagrees with ground_truth_bdecay; "
        "check the IA parent-chain walk and the hit-origin matching."
    )


# ---------------------------------------------------------------------------
# 2. Axis accuracy — grades the blind reconstruction against ANNI truth
# ---------------------------------------------------------------------------


def test_axis_angular_error(lor_harness, wandb_run):
    """Angular error between the blind LOR axis and the true ANNI emission
    axis. Graded on signal events with a complete photon pair — the
    population where the LOR is physically recoverable. A random axis would
    sit at 60 deg median (|cos| uniform); real reconstruction must do far
    better. Doppler acollinearity + position resolution set the floor, so we
    assert "clearly better than chance", not perfection.
    """
    t = lor_harness
    gradable = t["labels"] & t["has_complete_pair"] & np.isfinite(t["angular_errors"])
    errs = t["angular_errors"][gradable]
    assert errs.size > 0, "no gradable events (signal + complete pair + reconstructed axis)"

    median_err = float(np.median(errs))
    frac_within_10 = float((errs < 10.0).mean())

    # Context: errors on ALL signal events with an axis, including partial
    # deposits where the LOR is not physically recoverable.
    all_sig = t["labels"] & np.isfinite(t["angular_errors"])
    median_all = float(np.median(t["angular_errors"][all_sig])) if all_sig.any() else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs, bins=72, range=(0, 90), alpha=0.7, label=f"pair-complete (n={errs.size})")
    ax.axvline(median_err, color="tab:red", linestyle="--", label=f"median {median_err:.1f}°")
    ax.axvline(60.0, color="gray", linestyle=":", label="random-axis median (60°)")
    ax.set_xlabel("axis angular error [deg]")
    ax.set_ylabel("events")
    ax.set_title("Blind LOR axis vs ANNI truth")
    ax.legend()

    wandb_run.log(
        {
            "lor/axis/n_gradable": int(errs.size),
            "lor/axis/median_error_deg": median_err,
            "lor/axis/frac_within_10deg": frac_within_10,
            "lor/axis/median_error_all_signal_deg": median_all,
            "lor/axis/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert median_err < 45.0, (
        f"median blind-axis error is {median_err:.1f}° on pair-complete signal "
        "events — barely better than a random axis (60°). The chain "
        "bipartition or first-hit selection is not finding the annihilation "
        "geometry."
    )


# ---------------------------------------------------------------------------
# 3. Class overlap — same protocol as test_overlaps.py, on the new score
# ---------------------------------------------------------------------------


def test_lor_score_class_overlap(lor_harness, wandb_run):
    """Signal/background separability of the blind LOR score, measured
    exactly like the legacy feature overlaps (test_overlaps.py): common
    binning, class-conditional densities, Bhattacharyya + total variation.

    Scores are chi^2-like residuals >= 0 with a point mass near 0 for
    perfect events, so the histogram runs over log10(score + 1e-3).
    """
    t = lor_harness
    finite = np.isfinite(t["scores"])
    sig = t["scores"][finite & t["labels"]]
    bg = t["scores"][finite & ~t["labels"]]
    assert sig.size > 0 and bg.size > 0, "empty class after NaN filtering"

    sig_log = np.log10(sig + 1e-3)
    bg_log = np.log10(bg + 1e-3)
    lo = float(min(sig_log.min(), bg_log.min()))
    hi = float(max(sig_log.max(), bg_log.max()))
    edges = np.linspace(lo, hi if hi > lo else lo + 1e-6, 81)

    p_sig = _density(sig_log, edges)
    p_bg = _density(bg_log, edges)
    bc = _bhattacharyya(p_sig, p_bg, edges)
    tv = _total_variation(p_sig, p_bg, edges)

    nan_sig = int((~finite & t["labels"]).sum())
    nan_bg = int((~finite & ~t["labels"]).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stairs(p_sig, edges, label=f"true β (n={sig.size})", alpha=0.6, fill=True)
    ax.stairs(p_bg, edges, label=f"true bg (n={bg.size})", alpha=0.6, fill=True)
    ax.set_xlabel("log10(LOR score + 1e-3)")
    ax.set_ylabel("density")
    ax.set_title(f"Blind LOR score class-conditional overlap (BC={bc:.3f}, TV={tv:.3f})")
    ax.legend()

    wandb_run.log(
        {
            "overlap/lor_score/bhattacharyya": bc,
            "overlap/lor_score/total_variation": tv,
            "overlap/lor_score/n_beta": int(sig.size),
            "overlap/lor_score/n_bg": int(bg.size),
            "overlap/lor_score/n_beta_nan": nan_sig,
            "overlap/lor_score/n_bg_nan": nan_bg,
            "overlap/lor_score/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert tv > 0.05, (
        f"LOR-score total-variation distance between β and bg is only {tv:.3f} "
        "— the blind reconstruction's score distributions are nearly "
        "identical across classes; it adds no discriminative signal."
    )
