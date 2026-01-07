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
from physics.posterior import posterior_bdecay
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
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
):
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return False

    # Load event data onto GPU
    energies = torch.tensor(
        [event.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

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
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    for r in range(1, n_hits + 1):
        for c in itertools.combinations(range(n_hits), r):
            combos.append(c)
            sizes.append(r)

    max_r = max(sizes)
    n_combo = len(combos)

    # Pad combinations with -1
    idx = torch.full((n_combo, max_r), -1, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # Gather energies and positions
    valid_mask = idx >= 0

    energy_combo = torch.zeros((n_combo, max_r), dtype=energies.dtype, device=device)
    pos_combo = torch.zeros((n_combo, max_r, 3), dtype=positions.dtype, device=device)

    energy_combo[valid_mask] = energies[idx[valid_mask]]
    pos_combo[valid_mask] = positions[idx[valid_mask]]

    best_score = posterior_bdecay(
        energy_combo,
        pos_combo,
        ref_energy,
        tolerance,
        alpha_energy,
        alpha_arm,
        alpha_kn,
        sizes,
        n_combo,
        device,
    )

    return best_score


def annihilation_extractor_test_likelihoods(
    geometry_file: str,
    sim_file: str,
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    posteriors_bdecay = []
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

        _posterior_bdecay = detected_511_event_likelihoods(
            ref_energy,
            event,
            alpha_energy,
            alpha_arm,
            alpha_kn,
        )
        posteriors_bdecay.append(_posterior_bdecay)

    return posteriors_bdecay, ground_truths


##################################################################################


if __name__ == "__main__":
    load_dotenv()

    (
        posteriors_bdecay,
        ground_truths,
    ) = annihilation_extractor_test_likelihoods(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim", 1.0, 1.0, 0.0
    )
    posteriors_bdecay_true_events = []
    posteriors_bdecay_false_events = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            posteriors_bdecay_true_events.append(posteriors_bdecay[i])
        else:
            posteriors_bdecay_false_events.append(posteriors_bdecay[i])

    runs = [
        ("true_events_posterior_bdecay", posteriors_bdecay_true_events),
        ("false_events_posterior_bdecay", posteriors_bdecay_false_events),
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
