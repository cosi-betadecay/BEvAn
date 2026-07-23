from pathlib import Path

import pytest
from fake_megalib import (
    MPhysicalEvent,
    MString,
    PhysicalEvent,
    SimEvent,
    SimHit,
    SimIA,
    iter_sim_events,
    iter_tra_events,
    write_sim_file,
    write_tra_file,
)

from physics.ground_truths import anni_process_energies, ground_truth_bdecay

EXPECTED_SIM_EVENTS = 7182
EXPECTED_TRA_EVENTS = 4555
EXPECTED_TRA_COMPTON = 1851


def test_mstring_is_string_like() -> None:
    """Verify MString compares equal to plain strings and to matching MStrings."""
    assert MString("ANNI") == "ANNI"
    assert MString("ANNI") == MString("ANNI")
    assert MString("ANNI") != MString("COMP")


def test_sim_fixture_event_count(sim_path: Path) -> None:
    """Verify the reference ``.sim`` fixture parses to the expected event count."""
    assert sum(1 for _ in iter_sim_events(sim_path)) == EXPECTED_SIM_EVENTS


def test_sim_fixture_first_event_hand_checked(sim_path: Path) -> None:
    """Verify the first ``.sim`` event's hits, IAs, and derived truth labels."""
    event = next(iter_sim_events(sim_path))
    assert event.GetID() == 1
    assert event.GetNHTs() == 5
    assert event.GetNIAs() == 20

    hit = event.GetHTAt(0)
    assert hit.GetEnergy() == pytest.approx(9.71026)
    assert (hit.GetPosition().X(), hit.GetPosition().Y(), hit.GetPosition().Z()) == pytest.approx(
        (1.0175, -2.3125, 4.59798)
    )
    assert hit.IsOrigin(3) and hit.IsOrigin(11)
    assert not hit.IsOrigin(2)

    ia = event.GetIAAt(15)  # "IA ANNI 16;14;..."
    assert ia.GetProcess() == MString("ANNI")
    assert ia.GetId() == 16
    assert ia.GetOriginID() == 14

    # The two ANNI processes: one fully escaped (0 keV), one depositing
    # 274.60548 + 236.39343 = 510.99891 keV -> a beta+ event.
    assert anni_process_energies(event) == pytest.approx([0.0, 510.99891])
    assert ground_truth_bdecay(event)


def test_sim_fixture_has_both_classes(sim_path: Path) -> None:
    """Verify the reference ``.sim`` fixture contains both β⁺ and background events."""
    labels = [ground_truth_bdecay(e) for e in iter_sim_events(sim_path) if e.GetNHTs() > 0]
    assert 0 < sum(labels) < len(labels)


def test_sim_write_parse_round_trip(tmp_path: Path) -> None:
    """Verify events survive a ``.sim`` write/parse round trip, zero-hit events included."""
    events = [
        SimEvent(
            1,
            hits=[SimHit((1.0, -2.0, 3.0), 511.0, origins=(2, 3)), SimHit((0.0, 0.5, 0.0), 100.25)],
            ias=[SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1)],
        ),
        SimEvent(2),  # zero hits survives the round trip
        SimEvent(3, hits=[SimHit((0.0, 0.0, 0.0), 42.0)]),
    ]
    parsed = list(iter_sim_events(write_sim_file(tmp_path / "roundtrip.sim", events)))

    assert [e.GetID() for e in parsed] == [1, 2, 3]
    assert [e.GetNHTs() for e in parsed] == [2, 0, 1]
    first = parsed[0]
    assert first.GetNIAs() == 2
    assert first.GetIAAt(0).GetProcess() == "ANNI"
    assert first.GetIAAt(1).GetOriginID() == 1
    hit = first.GetHTAt(0)
    assert hit.GetEnergy() == pytest.approx(511.0)
    assert hit.GetPosition().Y() == pytest.approx(-2.0)
    assert hit.IsOrigin(2) and hit.IsOrigin(3) and not hit.IsOrigin(1)


def test_tra_fixture_counts_and_ids(tra_path: Path) -> None:
    """Verify the reference ``.tra`` fixture's event count, Compton count, and leading ids."""
    events = list(iter_tra_events(tra_path))
    assert len(events) == EXPECTED_TRA_EVENTS
    compton = [e for e in events if e.GetType() == MPhysicalEvent.c_Compton]
    assert len(compton) == EXPECTED_TRA_COMPTON
    assert [e.GetId() for e in events[:6]] == [3, 5, 7, 8, 9, 10]


def test_tra_compton_events_carry_unit_directions(tra_path: Path) -> None:
    """Verify every Compton event parsed from the ``.tra`` fixture carries a unit direction."""
    for event in iter_tra_events(tra_path):
        if event.GetType() == MPhysicalEvent.c_Compton:
            assert sum(c * c for c in event.direction) == pytest.approx(1.0)


def test_tra_write_parse_round_trip(tmp_path: Path) -> None:
    """Verify events survive a ``.tra`` write/parse round trip with normalized directions."""
    events = [
        PhysicalEvent(1, MPhysicalEvent.c_Compton, (0.0, 3.0, 4.0)),
        PhysicalEvent(2, MPhysicalEvent.c_Photo),
    ]
    parsed = list(iter_tra_events(write_tra_file(tmp_path / "roundtrip.tra", events)))

    assert [e.GetId() for e in parsed] == [1, 2]
    assert parsed[0].GetType() == MPhysicalEvent.c_Compton
    assert parsed[0].direction == pytest.approx((0.0, 0.6, 0.8))
    assert parsed[1].GetType() != MPhysicalEvent.c_Compton
