import itertools
import os
import sys
from typing import Any

import ROOT as M
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mathematics.calculations import calculate_tolerance
from physics.annihilation_detection import process
from physics.likelihoods import (
    compton_kinematic_likelihood,
    energy_likelihood,
    maximum_interaction_distance_likelihood,
)
from utils.reader_extraction import get_reader

##################################################################################
##################################################################################
##################################################################################


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

            wandb.log(
                {
                    "Energy Likelihood": _energy_likelihood.item(),
                    "Compton Kinematic Likelihood": _compton_kin_likelihood.item(),
                    "MID Likelihood": _mid_likelihood,
                }
            )


def annihilation_extractor_test_likelihoods(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        is_annihilation = process(event, ref_energy)

        if is_annihilation:
            detected_511_event_likelihoods(
                ref_energy,
                event,
            )


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)
    wandb.init(project="cosi-betadecay-likelihoods")

    annihilation_extractor_test_likelihoods(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )

    wandb.finish()
