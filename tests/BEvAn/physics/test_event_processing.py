import math

import pytest
import torch
from fake_megalib import SimEvent, SimHit

import physics.event_processing as ep
from physics.event_processing import (
    candidate_perms,
    energy_desc_order,
    event_data_processing,
    hit_subsets,
    orderings_for_size,
)


def _event(hits: list[tuple[tuple[float, float, float], float]]) -> SimEvent:
    """Build a single SimEvent from (position, energy) hit pairs."""
    return SimEvent(1, hits=[SimHit(pos, e) for pos, e in hits])


def test_candidate_perms_counts_and_uniqueness() -> None:
    """Enumerate unique hit-ordering permutations of the expected count."""
    p3 = candidate_perms(3)
    p4 = candidate_perms(4)
    assert p3.shape == (6, 3)
    assert p4.shape == (12, 4)
    assert len({tuple(row.tolist()) for row in p3}) == 6
    assert p3[0].tolist() == [0, 1, 2]


def test_hit_subsets_are_ascending_combinations() -> None:
    """Enumerate hit subsets as ascending-index combinations."""
    s = hit_subsets(5, 2)
    assert s.shape == (math.comb(5, 2), 2)
    assert torch.all(s[:, 0] < s[:, 1])
    assert s[0].tolist() == [0, 1]


def test_energy_desc_order_stable_on_ties() -> None:
    """Order hits by descending energy with a stable tie-break."""
    subsets = torch.tensor([[0, 1, 2]])
    assert energy_desc_order(subsets, torch.tensor([1.0, 3.0, 2.0])).tolist() == [[1, 2, 0]]
    # Equal energies keep ascending-index order (stable sort).
    assert energy_desc_order(subsets, torch.tensor([2.0, 2.0, 2.0])).tolist() == [[0, 1, 2]]


def test_orderings_for_size_dispatch() -> None:
    """Dispatch subset orderings by size and strategy (ckd vs energy)."""
    energies = torch.tensor([1.0, 3.0, 2.0])
    positions = torch.eye(3)
    singles = torch.tensor([[0], [1], [2]])
    assert torch.equal(orderings_for_size(singles, energies, positions, "ckd"), singles)

    pairs = torch.tensor([[0, 1]])
    assert orderings_for_size(pairs, energies, positions, "ckd").tolist() == [[1, 0]]

    triples = torch.tensor([[0, 1, 2]])
    assert orderings_for_size(triples, energies, positions, "energy").tolist() == [[1, 2, 0]]
    ckd = orderings_for_size(triples, energies, positions, "ckd")
    assert 1 <= ckd.shape[0] <= 6
    assert ckd.shape[1] == 3


def test_event_data_processing_empty_event() -> None:
    """Return all-None outputs for an event with no hits."""
    assert event_data_processing(SimEvent(1)) == (None, None, None)


def test_event_data_processing_shapes_and_padding() -> None:
    """Check output shapes and NaN padding of processed event tensors."""
    energies, positions, sizes = event_data_processing(
        _event([((0.0, 0.0, 0.0), 100.0), ((1.0, 0.0, 0.0), 200.0), ((0.0, 1.0, 0.0), 300.0)])
    )
    n = energies.shape[0]
    assert positions.shape == (n, 3, 3)
    assert sizes.shape == (n,)
    assert set(sizes.tolist()) == {1, 2, 3}
    assert (sizes == 1).sum() == 3
    assert (sizes == 2).sum() == 3  # C(3,2) subsets, one E1>=E2 ordering each

    for row in range(n):
        r = int(sizes[row])
        assert torch.isfinite(energies[row, :r]).all()
        assert torch.isnan(energies[row, r:]).all()
        assert torch.isnan(positions[row, r:, :]).all()

    hit_energies = {100.0, 200.0, 300.0}
    finite = energies[torch.isfinite(energies)]
    assert set(finite.tolist()) <= hit_energies


def test_event_data_processing_size_2_orders_by_energy() -> None:
    """Order size-2 subsets by descending energy."""
    energies, _, sizes = event_data_processing(_event([((0.0, 0.0, 0.0), 100.0), ((1.0, 0.0, 0.0), 200.0)]))
    pair = energies[sizes == 2][0]
    assert pair[0].item() >= pair[1].item()


def test_event_data_processing_truncates_to_highest_energy_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncate to the highest-energy hits when MAX_HITS is exceeded."""
    monkeypatch.setattr(ep, "MAX_HITS", 4)
    hits = [((float(i), 0.0, 0.0), 100.0 * (i + 1)) for i in range(6)]
    energies, _, sizes = event_data_processing(_event(hits))
    assert int(sizes.max()) == 4
    finite = set(energies[torch.isfinite(energies)].tolist())
    assert finite == {300.0, 400.0, 500.0, 600.0}


def test_ckd_residual_nan_without_internal_scatter() -> None:
    """Return NaN CKD residual when the sequence has no internal scatter."""
    energies = torch.tensor([[255.5, 255.5]])
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    assert math.isnan(ep.ckd_residual_batched(energies, positions).item())


def test_ckd_residual_zero_for_consistent_sequence() -> None:
    """Return zero CKD residual for a kinematically consistent sequence."""
    # Build a 3-hit sequence whose geometric scatter angle matches kinematics:
    # W_in = (E2+E3)/511, W_out = E3/511 with E2 = E3 = 255.5 gives
    # cos_kin = 1 + 511/511 - 511/255.5 = 0 -> a 90 degree internal scatter.
    energies = torch.tensor([[100.0, 255.5, 255.5]])
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
    assert ep.ckd_residual_batched(energies, positions).item() == pytest.approx(0.0, abs=1e-9)
