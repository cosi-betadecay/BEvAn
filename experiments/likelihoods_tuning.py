"""
Experiment to tune likelihood calculations for annihilation event detection.
Need to create some plots, etc. for analysis.
Combine the ground truth algorithm with the itertools permutations to check effectiveness of likelihoods
for different interaction orderings during real annihilation events.
"""

import itertools
from typing import Any

import ROOT as M
import torch
import tqdm

from mathematics.calculations import calculate_tolerance
from physics.annihilation_detection import process
from physics.likelihoods import (
    compton_kinematic_likelihood,
    energy_likelihood,
    maximum_interaction_distance_likelihood,
)
from utils.reader_extraction import get_reader


def detected_511_event_likelihoods(
    ref_energy: float,
    event: Any,
) -> bool:
    """Check if event contains a hit combination summing to the reference energy.

    Args:
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        Event (Any): MEGAlib event object with detector hits.

    Returns:
        bool: True if a hit combination matches the reference energy, else False.
    """
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    energies = torch.tensor([event.GetHTAt(i).GetEnergy() for i in range(n_hits)], dtype=torch.float32)
    positions = torch.tensor(
        [
            (
                event.GetHTAt(i).GetPosition().X(),
                event.GetHTAt(i).GetPosition().Y(),
                event.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
    )

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _energy_likelihood = energy_likelihood(energy_combo, ref_energy, tolerance)
            _compton_kin_likelihood = compton_kinematic_likelihood(energy_combo)
            _mid_likelihood = maximum_interaction_distance_likelihood(pos_combo)

            print(
                f"Energy Likelihood: {_energy_likelihood.item():.6f}, "
                f"Compton Kinematic Likelihood: {_compton_kin_likelihood.item():.6f}, "
                f"MID Likelihood: {_mid_likelihood:.6f}"
            )


def annihilation_extractor(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    """Extract annihilation events and compute detection performance metrics.

    Args:
        geometry_file (str): Path to the detector geometry setup file (XML).
        sim_file (str): Path to the MEGAlib simulation file (.sim).
        ref_energy (float): Reference energy to compare against (default 511 keV).

    Returns:
        None
    """
    reader = get_reader(geometry_file, sim_file)

    tp, fp, fn, tn = 0, 0, 0, 0

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        is_annihilation = process(event, ref_energy)

        if is_annihilation:
            detected_511 = detected_511_event_likelihoods(
                ref_energy,
                event,
            )
            break
