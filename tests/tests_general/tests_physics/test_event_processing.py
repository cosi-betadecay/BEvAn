"""Unit tests for physics/event_processing.py."""

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


def test_event_data_processing_returns_padded_combinations_with_sizes():
    """
    event_data_processing should return every non-empty hit subset padded to the
    maximum subset length, along with an integer tensor of subset sizes.
    """
    event = _FakeEvent(
        [
            _FakeHit(10.0, (1.0, 0.0, 0.0)),
            _FakeHit(20.0, (0.0, 1.0, 0.0)),
            _FakeHit(30.0, (0.0, 0.0, 1.0)),
        ]
    )

    energies, positions, sizes = event_data_processing(event)

    assert energies.shape == (7, 3)
    assert positions.shape == (7, 3, 3)
    assert sizes.shape == (7,)

    expected_energy_rows = {
        (10.0,),
        (20.0,),
        (30.0,),
        (10.0, 20.0),
        (10.0, 30.0),
        (20.0, 30.0),
        (10.0, 20.0, 30.0),
    }
    actual_energy_rows = {
        tuple(energy_row[:size].tolist())
        for energy_row, size in zip(energies.cpu(), sizes.cpu().tolist(), strict=False)
    }

    assert actual_energy_rows == expected_energy_rows
    seq = torch.arange(energies.shape[1]).unsqueeze(0)
    valid = seq < sizes.cpu().unsqueeze(1)
    assert torch.all(torch.isnan(energies.cpu()[~valid])), "Padding entries should be NaN beyond each candidate size"
    assert torch.all(torch.isnan(positions.cpu()[~valid])), "Padded positions should be NaN beyond each candidate size"


def test_zero_hit_event_returns_none_pair():
    """Events with no hits should still return a triple of Nones."""
    event = _FakeEvent([])

    energies, positions, sizes = event_data_processing(event)

    assert energies is None
    assert positions is None
    assert sizes is None


