"""Unit tests for physics/event_processing.py."""

import torch

from physics.event_processing import event_data_processing, strip_padding_from_event


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
        (10.0,),
        (20.0,),
        (30.0,),
        (10.0, 20.0),
        (10.0, 30.0),
        (20.0, 30.0),
        (10.0, 20.0, 30.0),
    }
    actual_energy_rows = {
        tuple(energy_row[mask_row].tolist())
        for energy_row, mask_row in zip(energies.cpu(), mask.cpu(), strict=False)
    }

    assert actual_energy_rows == expected_energy_rows
    assert torch.all(torch.isnan(energies[~mask])), "Padding entries should be NaN where the mask is False"
    assert torch.all(torch.isnan(positions[~mask])), "Padded positions should be NaN where the mask is False"


def test_zero_hit_event_returns_none_pair():
    """Events with no hits should still return a triple of Nones."""
    event = _FakeEvent([])

    energies, positions, mask = event_data_processing(event)

    assert energies is None
    assert positions is None
    assert mask is None


def test_strip_padding_from_event_removes_nan_padded_hits():
    """The helper should return only the real hits from a single padded candidate row."""
    energies = torch.tensor([10.0, 20.0, float("nan")], dtype=torch.float32)
    positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )

    stripped_energies, stripped_positions = strip_padding_from_event(energies, positions)

    assert tuple(stripped_energies.tolist()) == (10.0, 20.0)
    assert stripped_positions.shape == (2, 3)
    assert tuple(stripped_positions[1].tolist()) == (0.0, 1.0, 0.0)


def test_strip_padding_from_event_supports_batched_candidate_input():
    """The helper should return a cleaned candidate list when given batched padded input."""
    energies = torch.tensor([[10.0, float("nan")]], dtype=torch.float32)
    positions = torch.tensor([[[1.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]]], dtype=torch.float32)

    stripped = strip_padding_from_event(energies, positions)

    assert len(stripped) == 1
    stripped_energies, stripped_positions = stripped[0]
    assert tuple(stripped_energies.tolist()) == (10.0,)
    assert stripped_positions.shape == (1, 3)
