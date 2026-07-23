import math

import pytest
from fake_megalib import SimEvent, SimHit, SimIA

from physics.ground_truths import (
    anni_process_energies,
    bdecay_label_score,
    calculate_tolerance,
    ground_truth_bdecay,
)


def test_calculate_tolerance_is_n_sigma_of_fwhm() -> None:
    """Compute the energy tolerance as n sigma of the detector FWHM."""
    sigma = 2.25 / 2.355
    assert calculate_tolerance() == pytest.approx(5 * sigma)
    assert calculate_tolerance(1) == pytest.approx(sigma)
    assert calculate_tolerance(0) == 0.0


def _event_with_anni(secondary_hit_energies: list[float]) -> SimEvent:
    """One ANNI (IA id 1) whose single secondary (IA id 2) caused the given hits."""
    ias = [SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1)]
    hits = [SimHit((0.0, 0.0, float(i)), e, origins=(2,)) for i, e in enumerate(secondary_hit_energies)]
    return SimEvent(1, hits=hits, ias=ias)


def test_anni_process_energies_sums_secondary_hits() -> None:
    """Sum an ANNI's secondary-hit energies into one entry."""
    event = _event_with_anni([300.0, 211.0])
    assert anni_process_energies(event) == pytest.approx([511.0])


def test_anni_process_energies_one_entry_per_anni() -> None:
    """Produce one summed energy entry per ANNI interaction."""
    ias = [SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1), SimIA("ANNI", 3, 0), SimIA("PHOT", 4, 3)]
    hits = [
        SimHit((0.0, 0.0, 0.0), 300.0, origins=(2,)),
        SimHit((0.0, 0.0, 1.0), 211.0, origins=(2,)),
        SimHit((0.0, 0.0, 2.0), 100.0, origins=(4,)),
    ]
    event = SimEvent(1, hits=hits, ias=ias)
    assert anni_process_energies(event) == pytest.approx([511.0, 100.0])


def test_anni_process_energies_counts_each_hit_once() -> None:
    """Count a hit once even when several secondaries of one ANNI cause it."""
    # A hit caused by two secondaries of the same ANNI must not be double-counted.
    ias = [SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1), SimIA("PHOT", 3, 1)]
    event = SimEvent(1, hits=[SimHit((0.0, 0.0, 0.0), 100.0, origins=(2, 3))], ias=ias)
    assert anni_process_energies(event) == pytest.approx([100.0])


def test_anni_process_energies_empty_without_anni() -> None:
    """Return no energies for an event without an ANNI interaction."""
    event = SimEvent(1, hits=[SimHit((0.0, 0.0, 0.0), 511.0)], ias=[SimIA("COMP", 1, 0)])
    assert anni_process_energies(event) == []


def test_ground_truth_bdecay_thresholds_at_the_window() -> None:
    """Label β⁺ decay by whether an ANNI energy falls in the 511 keV window."""
    tolerance = calculate_tolerance()
    assert ground_truth_bdecay(_event_with_anni([511.0]))
    assert ground_truth_bdecay(_event_with_anni([511.0 + 0.9 * tolerance]))
    assert not ground_truth_bdecay(_event_with_anni([511.0 + 1.1 * tolerance]))
    assert not ground_truth_bdecay(_event_with_anni([400.0]))


def test_ground_truth_bdecay_window_scales_with_n_std() -> None:
    """Scale the β⁺ decay acceptance window with the n_std setting."""
    event = _event_with_anni([511.0 + 2.0])  # inside 5 sigma, outside 1 sigma
    assert ground_truth_bdecay(event, n_std=5)
    assert not ground_truth_bdecay(event, n_std=1)


def test_bdecay_label_score_matches_the_label() -> None:
    """Match the β⁺ label to its score compared against the tolerance."""
    event = _event_with_anni([505.0])
    assert bdecay_label_score(event) == pytest.approx(6.0)
    for n_std in (1, 5, 10):
        expected = bdecay_label_score(event) < calculate_tolerance(n_std)
        assert ground_truth_bdecay(event, n_std=n_std) == expected


def test_bdecay_label_score_inf_without_anni() -> None:
    """Return an infinite β⁺ label score for an event without an ANNI."""
    event = SimEvent(1, hits=[SimHit((0.0, 0.0, 0.0), 511.0)])
    assert math.isinf(bdecay_label_score(event))
    assert not ground_truth_bdecay(event)
