import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_generator_config import (
    gen_200_8_false,
    gen_200_8_true,
    gen_1000_2_false,
    gen_1000_2_true,
    gen_100000_24_false,
    gen_100000_24_true,
    gen_2000000_4_false,
    gen_2000000_4_true,
)

from mathematics.calculations import calculate_tolerance
from physics.likelihoods.energy import energy_kernel_bdecay


# ----------- Specific Tests ----------- #
def test_energy_kernel_symmetry():
    ref = 511.0
    sigma = 5.0

    E_plus = torch.tensor([ref + 10.0])
    E_minus = torch.tensor([ref - 10.0])

    k_plus = energy_kernel_bdecay(E_plus, ref, sigma)
    k_minus = energy_kernel_bdecay(E_minus, ref, sigma)

    assert torch.isclose(k_plus, k_minus)


def test_energy_kernel_monotonic_falloff():
    ref = 511.0
    sigma = 5.0

    E_close = torch.tensor([ref + 2.0])
    E_far = torch.tensor([ref + 20.0])

    k_close = energy_kernel_bdecay(E_close, ref, sigma)
    k_far = energy_kernel_bdecay(E_far, ref, sigma)

    assert k_close > k_far


def test_energy_kernel_range():
    ref = 511.0
    sigma = 5.0

    energies = torch.linspace(400, 620, 100)
    k = energy_kernel_bdecay(energies, ref, sigma)

    assert torch.all(k > 0)
    assert torch.all(k <= 1.0)


def test_energy_kernel_sigma_scaling():
    ref = 511.0
    E = torch.tensor([ref + 10.0])

    k_small = energy_kernel_bdecay(E, ref, sigma_e=2.0)
    k_large = energy_kernel_bdecay(E, ref, sigma_e=10.0)

    assert k_large > k_small


def test_energy_kernel_batch_order_invariance():
    ref = 511.0
    sigma = 5.0

    energies = torch.tensor(
        [
            [200.0, 311.0],  # sum = 511
            [180.0, 331.0],  # sum = 511
        ]
    )

    k1 = energy_kernel_bdecay(energies, ref, sigma)
    k2 = energy_kernel_bdecay(energies.flip(0), ref, sigma)

    assert torch.allclose(k1.sort().values, k2.sort().values)


def test_energy_kernel_extreme_values():
    ref = 511.0
    sigma = 5.0

    E_very_far = torch.tensor([1e6])
    k = energy_kernel_bdecay(E_very_far, ref, sigma)

    assert k < 1e-6


def test_energy_kernel_has_gradient():  # For future ML purposes if applicable
    energies = torch.tensor([200.0, 311.0], requires_grad=True)

    k = energy_kernel_bdecay(energies, 511.0, 5.0)
    k.backward()

    assert energies.grad is not None


# ----------- General Tests ----------- #
# -- True Events Tests -- #
def test_1000_2_true_events_mean():
    energies, _ = gen_1000_2_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_1000_2_true_events_max():
    energies, _ = gen_1000_2_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_1000_2_true_events_min():
    energies, _ = gen_1000_2_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_2000000_4_true_events_mean():
    energies, _ = gen_2000000_4_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_2000000_4_true_events_max():
    energies, _ = gen_2000000_4_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_2000000_4_true_events_min():
    energies, _ = gen_2000000_4_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_200_8_true_events_mean():
    energies, _ = gen_200_8_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_200_8_true_events_max():
    energies, _ = gen_200_8_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_200_8_true_events_min():
    energies, _ = gen_200_8_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_100000_24_true_events_mean():
    energies, _ = gen_100000_24_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_100000_24_true_events_max():
    energies, _ = gen_100000_24_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_100000_24_true_events_min():
    energies, _ = gen_100000_24_true

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


# -- False Events Tests -- #
def test_1000_2_false_events_mean():
    energies, _ = gen_1000_2_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_1000_2_false_events_max():
    energies, _ = gen_1000_2_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_1000_2_false_events_min():
    energies, _ = gen_1000_2_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_2000000_4_false_events_mean():
    energies, _ = gen_2000000_4_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_2000000_4_false_events_max():
    energies, _ = gen_2000000_4_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_2000000_4_false_events_min():
    energies, _ = gen_2000000_4_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_200_8_false_events_mean():
    energies, _ = gen_200_8_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_200_8_false_events_max():
    energies, _ = gen_200_8_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_200_8_false_events_min():
    energies, _ = gen_200_8_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_100000_24_false_events_mean():
    energies, _ = gen_100000_24_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_100000_24_false_events_max():
    energies, _ = gen_100000_24_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_100000_24_false_events_min():
    energies, _ = gen_100000_24_false

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05
