import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthetic_data_generator import SyntheticDataGenerator

from mathematics.calculations import calculate_tolerance
from physics.likelihoods.energy import energy_kernel_bdecay


def test_energy_kernel_true_events_peak_near_511():
    gen = SyntheticDataGenerator(max_hits_per_sequence=8, seed=0)

    energies, _ = gen.generate(
        num_samples=200,
        mode=True,
    )

    # energies er allerede [N, max_hits]
    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    assert torch.isfinite(vals).all()
    assert vals.mean() > 0.8


def test_energy_kernel_false_events_peak_near_511():
    gen = SyntheticDataGenerator(max_hits_per_sequence=8, seed=0)

    energies, _ = gen.generate(
        num_samples=200,
        mode=False,
    )

    # energies er allerede [N, max_hits]
    vals = energy_kernel_bdecay(
        energies,
        ref_energy=511.0,
        sigma_e=calculate_tolerance(),
    )

    print(vals)

    assert torch.isfinite(vals).all()
    assert vals.mean() < 0.5
