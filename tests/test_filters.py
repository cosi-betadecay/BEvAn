import numpy as np
from src.beta_decay.physics.filters import compton_cone_filter


def test_compton_cone_filter_requires_two_hits() -> None:
    energies = [600.0]
    positions = [np.zeros(3)]

    assert compton_cone_filter(energies, positions) is False


def test_compton_cone_filter_rejects_low_energy_sum() -> None:
    energies = [200.0, 200.0]
    positions = [np.zeros(3), np.array([1.0, 0.0, 0.0])]

    assert compton_cone_filter(energies, positions) is False


def test_compton_cone_filter_rejects_out_of_range_angle() -> None:
    energies = [600.0, 50.0]
    positions = [np.zeros(3), np.array([1.0, 1.0, 0.0])]

    assert compton_cone_filter(energies, positions) is False


def test_compton_cone_filter_rejects_colocated_hits() -> None:
    energies = [426.54028436018956, 300.0]
    positions = [np.ones(3), np.ones(3)]

    assert compton_cone_filter(energies, positions) is False


def test_compton_cone_filter_accepts_valid_pair() -> None:
    energies = [426.54028436018956, 300.0, 100.0]
    positions = [
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
    ]

    assert compton_cone_filter(energies, positions) is True
