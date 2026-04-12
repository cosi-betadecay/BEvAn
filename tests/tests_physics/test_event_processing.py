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


def test_event_data_processing_returns_padded_combinations_with_mask():
    """
    event_data_processing should return every non-empty hit subset padded to the
    maximum subset length, along with a boolean mask that marks which entries
    are real hits versus zero-padding.
    """
    event = _FakeEvent(
        [
            _FakeHit(10.0, (1.0, 0.0, 0.0)),
            _FakeHit(20.0, (0.0, 1.0, 0.0)),
            _FakeHit(30.0, (0.0, 0.0, 1.0)),
        ]
    )

    energies, positions, mask = event_data_processing(event)

    assert energies.shape == (7, 3)
    assert positions.shape == (7, 3, 3)
    assert mask.shape == (7, 3)
    assert mask.dtype == torch.bool

    expected_energy_rows = {
        ((10.0, 0.0, 0.0), (True, False, False)),
        ((20.0, 0.0, 0.0), (True, False, False)),
        ((30.0, 0.0, 0.0), (True, False, False)),
        ((10.0, 20.0, 0.0), (True, True, False)),
        ((10.0, 30.0, 0.0), (True, True, False)),
        ((20.0, 30.0, 0.0), (True, True, False)),
        ((10.0, 20.0, 30.0), (True, True, True)),
    }
    actual_energy_rows = {
        (tuple(energy_row.tolist()), tuple(mask_row.tolist()))
        for energy_row, mask_row in zip(energies.cpu(), mask.cpu(), strict=False)
    }

    assert actual_energy_rows == expected_energy_rows
    assert torch.all(energies[~mask] == 0.0), "Padding entries should remain zero where the mask is False"


def test_zero_hit_event_returns_none_pair():
    """Events with no hits should still return a triple of Nones."""
    event = _FakeEvent([])

    energies, positions, mask = event_data_processing(event)

    assert energies is None
    assert positions is None
    assert mask is None
