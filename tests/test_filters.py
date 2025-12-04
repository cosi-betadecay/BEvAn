import torch
from src.beta_decay.physics.filters import verify_compton_angle


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
