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
from physics.annihilation_detection import ground_truth
from physics.likelihoods.compton_kin import compton_kinematic_angle_likelihood
from physics.likelihoods.energy import energy_likelihood
from physics.posterior import posterior
from utils.plots import (
    cdf,
    ecdf_slope,
    histogram,
    histogram_log,
    log_calculations,
    violin_plot,
)
from utils.reader_extraction import get_reader

##################################################################################


# Data extraction


def detected_511_event_likelihoods(
    ref_energy: float,
    event: Any,
):
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

    if len(energies) == 0:
        return 0.0, 0.0, 0.0

    _energy_likelihood = -float("inf")
    _compton_kinematic_weight = -float("inf")
    _posterior = -float("inf")

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _energy_likelihood = max(
                _energy_likelihood, energy_likelihood(energy_combo, ref_energy, tolerance)
            )
            _compton_kinematic_weight = max(
                _compton_kinematic_weight, compton_kinematic_angle_likelihood(energy_combo)
            )

            _posterior = max(
                _posterior, posterior(energy_combo, pos_combo, 0, ref_energy, tolerance, 1.0, 1.0, 0, 0, 0)
            )

    return _energy_likelihood, _compton_kinematic_weight, _posterior


def annihilation_extractor_test_likelihoods(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    energy_likelihoods = []
    compton_kinematic_weights = []
    posteriors = []
    ground_truths = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        is_annihilation = ground_truth(event, ref_energy)
        if is_annihilation:
            ground_truths.append(1)
        else:
            ground_truths.append(0)

        energy_likelihood, compton_kinematic_weight, posterior = detected_511_event_likelihoods(
            ref_energy,
            event,
        )
        energy_likelihoods.append(energy_likelihood)
        compton_kinematic_weights.append(compton_kinematic_weight)
        posteriors.append(posterior)

    return energy_likelihoods, compton_kinematic_weights, posteriors, ground_truths


##################################################################################


if __name__ == "__main__":
    load_dotenv()

    (
        energy_likelihoods,
        compton_kinematic_angle_weights,
        posteriors,
        ground_truths,
    ) = annihilation_extractor_test_likelihoods(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )
    energy_likelihoods_true_events = []
    energy_likelihoods_false_events = []
    compton_kinematic_true_events = []
    compton_kinematic_false_events = []
    posteriors_true_events = []
    posteriors_false_events = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            energy_likelihoods_true_events.append(energy_likelihoods[i])
            compton_kinematic_true_events.append(compton_kinematic_angle_weights[i])
            posteriors_true_events.append(posteriors[i])
        else:
            energy_likelihoods_false_events.append(energy_likelihoods[i])
            compton_kinematic_false_events.append(compton_kinematic_angle_weights[i])
            posteriors_false_events.append(posteriors[i])
    runs = [
        ("true_events_energy_likelihood", energy_likelihoods_true_events),
        ("false_events_energy_likelihood", energy_likelihoods_false_events),
        ("true_events_compton_kinematic", compton_kinematic_true_events),
        ("false_events_compton_kinematic", compton_kinematic_false_events),
        ("true_events_posterior", posteriors_true_events),
        ("false_events_posterior", posteriors_false_events),
    ]

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-likelihoods", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

        wandb.finish()
