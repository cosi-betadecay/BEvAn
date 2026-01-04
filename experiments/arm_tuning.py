import itertools
import os
import sys
from typing import Any

import ROOT as M
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mathematics.calculations import calculate_tolerance
from physics.annihilation_detection import ground_truth
from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_pdf_bdecay
from physics.likelihoods.kn import klein_nishina_pdf
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


def detected_511_event_likelihoods(
    ref_energy: float,
    event: Any,
):
    n_hits = event.GetNHTs()
    tolerance = calculate_tolerance()

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
        return 0.0

    _best = -float("inf")

    for r in range(3, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _arm = angular_resolution_measure_kernel(energy_combo, pos_combo)
            _kn = klein_nishina_pdf(energy_combo)
            _energy = energy_pdf_bdecay(energy_combo, ref_energy, tolerance)

            _arm = torch.clamp(_arm, min=1e-12)
            _kn = torch.clamp(_kn, min=1e-12)
            _energy = torch.clamp(_energy, min=1e-12)

            score = torch.log(_arm) + torch.log(_energy) + 0.5 * torch.log(_kn)  # alpha = 0.5

            _best = max(_best, score.item())

    return _best


def annihilation_extractor_test_kn(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    combs = []
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

        _comb = detected_511_event_likelihoods(
            ref_energy,
            event,
        )

        if _comb == -float("inf"):
            _comb = 0.0

        combs.append(_comb)

    return combs, ground_truths


##################################################################################


if __name__ == "__main__":
    load_dotenv()

    (
        combs,
        ground_truths,
    ) = annihilation_extractor_test_kn(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )

    true_events_arm_kn_energies = []
    false_events_arm_kn_energies = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            true_events_arm_kn_energies.append(combs[i])
        else:
            false_events_arm_kn_energies.append(combs[i])

    runs = [
        ("arm-kn-energy", combs),
        ("arm-kn-energy_true_events", true_events_arm_kn_energies),
        ("arm-kn-energy_false_events", false_events_arm_kn_energies),
    ]

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-arm", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

        wandb.finish()
