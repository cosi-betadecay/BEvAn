import math

import pytest
import torch

from physics.ground_truths import calculate_tolerance
from physics.physics_factors import (
    annihilation_angle,
    arm,
    arm_fixed_size,
    delta_E,
    per_subset_back_to_back,
    theta_geo,
    theta_kin,
)

Z_HAT = torch.tensor([0.0, 0.0, 1.0])


def test_theta_geo_right_angle():
    # First leg (0,0,0) -> (0,1,0) is perpendicular to the incoming z direction.
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    assert theta_geo(positions, Z_HAT).item() == pytest.approx(math.pi / 2)


def test_theta_geo_parallel_and_antiparallel():
    along = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    against = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]]])
    assert theta_geo(along, Z_HAT).item() == pytest.approx(0.0)
    assert theta_geo(against, Z_HAT).item() == pytest.approx(math.pi)


def test_theta_kin_half_absorption_is_right_angle():
    # E1 = 511, E2 = 255.5: cos = 1 - 511 (1/255.5 - 1/511) = 0.
    assert theta_kin(torch.tensor([255.5, 255.5])).item() == pytest.approx(math.pi / 2)


def test_theta_kin_batched_matches_scalar():
    energies = torch.tensor([[255.5, 255.5], [400.0, 111.0]])
    batched = theta_kin(energies)
    for i in range(2):
        assert batched[i].item() == pytest.approx(theta_kin(energies[i]).item())


def test_delta_E_min_over_subsets_ignores_padding():
    nan = float("nan")
    energies = torch.tensor([[400.0, nan], [300.0, 211.0]])
    sizes = torch.tensor([1, 2])
    assert delta_E(energies, sizes).item() == pytest.approx(0.0)
    assert delta_E(energies[:1], sizes[:1]).item() == pytest.approx(111.0)


def test_arm_fixed_size_zero_for_perfect_511_subset():
    # Geometry gives theta_geo = pi/2 and 255.5 + 255.5 keV gives theta_kin = pi/2,
    # with the energy sum exactly on 511 -> no push. Feature is exactly 0.
    energies = torch.tensor([[255.5, 255.5]])
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    assert arm_fixed_size(energies, positions, Z_HAT).item() == pytest.approx(0.0, abs=1e-6)


def test_arm_fixed_size_energy_push_direction():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    tolerance = calculate_tolerance()
    excess = 20.0
    energies = torch.tensor([[255.5 + excess, 255.5]])
    feature = arm_fixed_size(energies, positions, Z_HAT).item()
    angular = ((theta_geo(positions, Z_HAT) - theta_kin(energies)) / math.pi).item()
    assert feature == pytest.approx(angular + (excess - tolerance) / 511.0, abs=1e-6)


def test_arm_fixed_size_nan_for_single_hit():
    assert math.isnan(arm_fixed_size(torch.tensor([[511.0]]), torch.zeros(1, 1, 3), Z_HAT).item())


def test_arm_pools_best_across_sizes():
    nan = float("nan")
    energies = torch.tensor([[511.0, nan], [255.5, 255.5]])
    positions = torch.tensor([[[0.0, 0.0, 0.0], [nan, nan, nan]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    sizes = torch.tensor([1, 2])
    assert arm(energies, positions, Z_HAT, sizes).item() == pytest.approx(0.0, abs=1e-6)


def test_arm_nan_when_no_size_2_subset():
    energies = torch.tensor([[511.0]])
    positions = torch.zeros(1, 1, 3)
    assert math.isnan(arm(energies, positions, Z_HAT, torch.tensor([1])).item())


def _back_to_back_subset() -> tuple[torch.Tensor, torch.Tensor]:
    # Vertex at the origin with two 511 keV arms in exactly opposite directions.
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    energies = torch.tensor([[10.0, 511.0, 511.0]])
    return positions, energies


def test_per_subset_back_to_back_perfect_pair():
    positions, energies = _back_to_back_subset()
    assert per_subset_back_to_back(positions, energies).item() == pytest.approx(-1.0)


def test_per_subset_back_to_back_gate_on_arm_energy():
    positions, _ = _back_to_back_subset()
    weak = torch.tensor([[10.0, 100.0, 100.0]])  # arms far below the 511 floor
    assert math.isinf(per_subset_back_to_back(positions, weak).item())


def test_per_subset_back_to_back_energy_push_off_target():
    positions, _ = _back_to_back_subset()
    tolerance = calculate_tolerance()
    excess = 200.0
    energies = torch.tensor([[10.0, 511.0, 511.0 + excess]])
    expected = -1.0 + 0.3 * (excess - 2.0 * tolerance) / 511.0
    assert per_subset_back_to_back(positions, energies).item() == pytest.approx(expected, abs=1e-5)


def test_annihilation_angle_best_over_subsets():
    positions, energies = _back_to_back_subset()
    sizes = torch.tensor([3])
    assert annihilation_angle(positions, energies, sizes).item() == pytest.approx(-1.0)


def test_annihilation_angle_nan_below_size_3():
    positions = torch.zeros(2, 2, 3)
    energies = torch.full((2, 2), 511.0)
    assert math.isnan(annihilation_angle(positions, energies, torch.tensor([1, 2])).item())
