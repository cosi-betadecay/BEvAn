"""Unit tests for physics/event_processing.py."""

import math

import torch

from physics.event_processing import event_data_processing, iter_event_permutation_batches


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


def test_event_data_processing_returns_base_hit_tensors():
    """
    event_data_processing should return the original event hits once, without
    materializing the full permutation search space in memory.
    """
    event = _FakeEvent(
        [
            _FakeHit(10.0, (1.0, 0.0, 0.0)),
            _FakeHit(20.0, (0.0, 1.0, 0.0)),
            _FakeHit(30.0, (0.0, 0.0, 1.0)),
        ]
    )

    energies, positions = event_data_processing(event)

    assert energies.shape == (3,)
    assert positions.shape == (3, 3)
    assert tuple(energies.cpu().tolist()) == (10.0, 20.0, 30.0)
    assert tuple(positions[0].cpu().tolist()) == (1.0, 0.0, 0.0)


def test_permutation_batches_cover_all_full_length_orderings_without_padding():
    """
    iter_event_permutation_batches must enumerate the full N! ordering search
    space while keeping each emitted batch bounded in size.
    """
    base_energies = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    base_positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    batches = list(iter_event_permutation_batches(base_energies, base_positions, batch_size=2))

    assert len(batches) == math.ceil(math.factorial(3) / 2)

    expected_energy_rows = {
        (10.0, 20.0, 30.0),
        (10.0, 30.0, 20.0),
        (20.0, 10.0, 30.0),
        (20.0, 30.0, 10.0),
        (30.0, 10.0, 20.0),
        (30.0, 20.0, 10.0),
    }
    actual_energy_rows = {
        tuple(row.tolist()) for energy_batch, _ in batches for row in energy_batch.cpu()
    }

    assert actual_energy_rows == expected_energy_rows
    for energy_batch, position_batch in batches:
        assert energy_batch.shape[1] == 3
        assert position_batch.shape[1:] == (3, 3)
        assert torch.all(energy_batch > 0.0), "Found padded zero-energy pseudo-hits in permutation batches"


def test_zero_hit_event_returns_none_pair():
    """Events with no hits should still return (None, None)."""
    event = _FakeEvent([])

    energies, positions = event_data_processing(event)

    assert energies is None
    assert positions is None
