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
from physics.likelihoods.klein_nishina_likelihood import klein_nishina_likelihood
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
        return 0.0

    _kn = -float("inf")

    for r in range(2, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _kn = max(_kn, klein_nishina_likelihood(energy_combo))

    return _kn


def annihilation_extractor_test_kn(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    kns = []
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

        kn = detected_511_event_likelihoods(
            ref_energy,
            event,
        )
        kns.append(kn)

    return kns, ground_truths


##################################################################################


if __name__ == "__main__":
    load_dotenv()

    (
        kns,
        ground_truths,
    ) = annihilation_extractor_test_kn(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )
    kns_true_events = []
    kns_false_events = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            kns_true_events.append(kns[i])
        else:
            kns_false_events.append(kns[i])
    runs = [
        ("true_events_kn", kns_true_events),
        ("false_events_kn", kns_false_events),
    ]

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-kn", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

        wandb.finish()
