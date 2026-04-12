"""Unit tests for physics/event_processing.py."""

import math

import torch

from physics.event_processing import event_data_processing


class _FakePosition:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = x
        self._y = y
        self._z = z

    def X(self) -> float:
        return self._x

    def Y(self) -> float:
        return self._y

    def Z(self) -> float:
        return self._z


class _FakeHit:
    def __init__(self, energy: float, position: tuple[float, float, float]) -> None:
        self._energy = energy
        self._position = _FakePosition(*position)

    def GetEnergy(self) -> float:
        return self._energy

    def GetPosition(self) -> _FakePosition:
        return self._position


class _FakeEvent:
    def __init__(self, hits: list[_FakeHit]) -> None:
        self._hits = hits

    def GetNHTs(self) -> int:
        return len(self._hits)

    def GetHTAt(self, index: int) -> _FakeHit:
        return self._hits[index]


def test_returns_all_full_length_permutations_only():
    """
    For an N-hit event, event_data_processing must return all N! orderings of
    the full hit list and no shorter, zero-padded subsets.
    """
    event = _FakeEvent(
        [
            _FakeHit(10.0, (1.0, 0.0, 0.0)),
            _FakeHit(20.0, (0.0, 1.0, 0.0)),
            _FakeHit(30.0, (0.0, 0.0, 1.0)),
        ]
    )

    energies, positions = event_data_processing(event)

    assert energies.shape == (math.factorial(3), 3)
    assert positions.shape == (math.factorial(3), 3, 3)

    expected_energy_rows = {
        (10.0, 20.0, 30.0),
        (10.0, 30.0, 20.0),
        (20.0, 10.0, 30.0),
        (20.0, 30.0, 10.0),
        (30.0, 10.0, 20.0),
        (30.0, 20.0, 10.0),
    }
    actual_energy_rows = {tuple(row.tolist()) for row in energies.cpu()}

    assert actual_energy_rows == expected_energy_rows
    assert torch.all(energies > 0.0), "Found padded zero-energy pseudo-hits in full-length permutations"


def test_zero_hit_event_returns_none_pair():
    """Events with no hits should still return (None, None)."""
    event = _FakeEvent([])

    energies, positions = event_data_processing(event)

    assert energies is None
    assert positions is None
