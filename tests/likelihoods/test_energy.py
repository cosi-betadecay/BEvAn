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


# -- True Events Tests -- #
def test_1000_2_true_events_mean():
    gen = gen_1000_2_true

    energies, _ = gen.generate(
        num_samples=1000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_1000_2_true_events_max():
    gen = gen_1000_2_true

    energies, _ = gen.generate(
        num_samples=1000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_1000_2_true_events_min():
    gen = gen_1000_2_true

    energies, _ = gen.generate(
        num_samples=1000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_2000000_4_true_events_mean():
    gen = gen_2000000_4_true

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_2000000_4_true_events_max():
    gen = gen_2000000_4_true

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_2000000_4_true_events_min():
    gen = gen_2000000_4_true

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_200_8_true_events_mean():
    gen = gen_200_8_true

    energies, _ = gen.generate(
        num_samples=200,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_200_8_true_events_max():
    gen = gen_200_8_true

    energies, _ = gen.generate(
        num_samples=200,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_200_8_true_events_min():
    gen = gen_200_8_true

    energies, _ = gen.generate(
        num_samples=200,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


def test_100000_24_true_events_mean():
    gen = gen_100000_24_true

    energies, _ = gen.generate(
        num_samples=100000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) > 0.8


def test_100000_24_true_events_max():
    gen = gen_100000_24_true

    energies, _ = gen.generate(
        num_samples=100000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) > 0.99


def test_100000_24_true_events_min():
    gen = gen_100000_24_true

    energies, _ = gen.generate(
        num_samples=100000,
        mode=1,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) > 0.5


# -- False Events Tests -- #
def test_1000_2_false_events_mean():
    gen = gen_1000_2_false

    energies, _ = gen.generate(
        num_samples=1000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_1000_2_false_events_max():
    gen = gen_1000_2_false

    energies, _ = gen.generate(
        num_samples=1000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_1000_2_false_events_min():
    gen = gen_1000_2_false

    energies, _ = gen.generate(
        num_samples=1000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_2000000_4_false_events_mean():
    gen = gen_2000000_4_false

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_2000000_4_false_events_max():
    gen = gen_2000000_4_false

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_2000000_4_false_events_min():
    gen = gen_2000000_4_false

    energies, _ = gen.generate(
        num_samples=2000000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_200_8_false_events_mean():
    gen = gen_200_8_false

    energies, _ = gen.generate(
        num_samples=200,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_200_8_false_events_max():
    gen = gen_200_8_false

    energies, _ = gen.generate(
        num_samples=200,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_200_8_false_events_min():
    gen = gen_200_8_false

    energies, _ = gen.generate(
        num_samples=200,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05


def test_100000_24_false_events_mean():
    gen = gen_100000_24_false

    energies, _ = gen.generate(
        num_samples=100000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.mean(vals) < 0.2


def test_100000_24_false_events_max():
    gen = gen_100000_24_false

    energies, _ = gen.generate(
        num_samples=100000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.max(vals) < 0.3


def test_100000_24_false_events_min():
    gen = gen_100000_24_false

    energies, _ = gen.generate(
        num_samples=100000,
        mode=2,
    )

    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert torch.min(vals) < 0.05
