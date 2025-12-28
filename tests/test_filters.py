import numpy as np
import torch

from physics.filters import angular_resolution_measure_filter, verify_compton_angle

# ----------------------------
# Tests: Compton kinematic angle filter
# ----------------------------


def test_verify_compton_angle_valid():
    energies = torch.tensor([300.0, 200.0])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_invalid():
    energies = torch.tensor([50.0, 10.0])
    assert verify_compton_angle(energies) is False


def test_verify_compton_angle_one_energy():
    energies = torch.tensor([511.5])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_high_precision():
    energies = torch.tensor([255.5, 255.5])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_grazing_scatter():
    assert verify_compton_angle(torch.tensor([5.0, 506.0])) is True


def test_verify_compton_angle_backscatter():
    assert verify_compton_angle(torch.tensor([340.0, 171.0])) is True


def test_verify_compton_angle_overdeposit():
    assert verify_compton_angle(torch.tensor([500.0, 11.0])) is False


def test_verify_compton_angle_resolution_margin():
    # Slightly inconsistent but pushed inside sigma
    assert verify_compton_angle(torch.tensor([300.0, 211.0])) is True


def test_verify_compton_angle_max_single_scatter():
    energies = torch.tensor([340.0, 171.0])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_ultra_grazing():
    energies = torch.tensor([0.5, 510.0])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_zero_energy_deposit():
    energies = torch.tensor([0.0, 511.0])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_zero_remaining_energy():
    energies = torch.tensor([511.0, 0.0])
    assert verify_compton_angle(energies) is False


def test_verify_compton_angle_negative_energy():
    energies = torch.tensor([-10.0, 520.0])
    assert verify_compton_angle(energies) is False


def test_verify_compton_angle_floating_point_edge():
    energies = torch.tensor([339.9999, 171.0001])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_tiny_energies_unphysical():
    energies = torch.tensor([0.01, 0.02])
    assert verify_compton_angle(energies) is False


def test_verify_compton_angle_three_hits():
    energies = torch.tensor([200.0, 200.0, 111.0])
    assert verify_compton_angle(energies) is True


def test_verify_compton_angle_machine_precision_values():
    energies = torch.tensor([1e-8, 1e-8])
    assert verify_compton_angle(energies) is False


# ----------------------------
# Tests: ARM filter
# ----------------------------


def test_arm_not_enough_hits_one():
    energies = torch.tensor([300.0])
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is None


def test_arm_not_enough_hits_two():
    energies = torch.tensor([300.0, 200.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is None


def test_arm_invalid_straight_forward():
    energies = torch.tensor([230.0, 220.0, 11.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_invalid_small_difference():
    energies = torch.tensor([260.0, 200.0, 51.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_large_difference():
    # theta_geo ≠ theta_kin → ARM large → False
    energies = torch.tensor([400.0, 50.0, 61.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_threshold_edge_exact():
    energies = torch.tensor([300.0, 200.0, 11.0], dtype=torch.float32)
    positions = torch.tensor([[0, 0, 0], [1, 0, 0], [np.cos(1.465), np.sin(1.465), 0]], dtype=torch.float32)
    assert angular_resolution_measure_filter(energies, positions) is True


def test_arm_threshold_barely_above():
    # ARM slightly above threshold → False
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.35, 0.0]])  # Slightly larger offset
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_geo_degenerate_zero_vector():
    # x0 == x1 → v_in = 0 → θ_geo = None → False
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_geo_degenerate_double_zero():
    # x1 == x2 → v_out = 0 → θ_geo = None → False
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_geo_unphysical_cosine():
    # cos(theta) > 1.05 → geometric angle invalid → False
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.00001, 0.0, 0.0], [0.00002, 0.0, 0.0]]
    )  # Almost identical points
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_kin_unphysical_negative_energy():
    energies = torch.tensor([-10.0, 520.0, 1.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_kin_unphysical_zero_sum():
    energies = torch.tensor([0.0, 0.0, 0.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_kin_cosine_exceeds_limit():
    # Energies giving cos(theta_kin) far > 1.05
    energies = torch.tensor([500.0, 1.0, 1.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_valid_kinematics_invalid_geometry():
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # geometry broken
            [1.0, 0.0, 0.0],
        ]
    )
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_backscatter_true():
    # Rough 180° geometric & kinematic match
    energies = torch.tensor([341.0, 170.0, 0.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # backscatter geometry
    assert angular_resolution_measure_filter(energies, positions) is True


def test_arm_forwardscatter_mismatch():
    energies = torch.tensor([300.0, 150.0, 61.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0]])  # geometric angle large
    assert angular_resolution_measure_filter(energies, positions) is False


def test_arm_floating_point_noise():
    energies = torch.tensor([300.0, 200.0, 11.0])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1e-6, 0.0],  # slight jitter
            [2.0, 0.0, 0.0],
        ]
    )
    assert angular_resolution_measure_filter(energies, positions) in (True, False)
    # This test ensures no crash & no None
