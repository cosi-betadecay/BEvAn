import itertools
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ROOT as M
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mathematics.calculations import calculate_tolerance
from physics.likelihoods import (
    energy_likelihood,
    klein_nishina_weight,
)
from utils.reader_extraction import get_reader

##################################################################################
##################################################################################
##################################################################################


# Data extraction


def detected_511_event_likelihoods(
    ref_energy: float,
    event: Any,
) -> tuple[torch.Tensor, torch.Tensor, float]:
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

    _energy_likelihood = -float("inf")
    _klein_nishina_weight = -float("inf")

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _energy_likelihood = max(
                _energy_likelihood, energy_likelihood(energy_combo, ref_energy, tolerance)
            )
            _klein_nishina_weight = max(_klein_nishina_weight, klein_nishina_weight(energy_combo))

    return _energy_likelihood, _klein_nishina_weight


def annihilation_extractor_test_likelihoods(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    energy_likelihoods = []
    klein_nishina_weight_likelihoods = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        # is_annihilation = process(event, ref_energy)

        energy_likelihood, klein_nishina_weight_likelihood = detected_511_event_likelihoods(
            ref_energy,
            event,
        )
        energy_likelihoods.append(energy_likelihood)
        klein_nishina_weight_likelihoods.append(klein_nishina_weight_likelihood)

    return energy_likelihoods, klein_nishina_weight_likelihoods


##################################################################################


# Plots


def as_np(vals) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.clip(arr, 0.0, 1.0)


def _log_image(key: str, fig: plt.Figure) -> None:
    # Log image and pin to summary so it shows up immediately on the dashboard.
    img = wandb.Image(fig)
    wandb.log({key: img})
    if wandb.run is not None:
        wandb.run.summary[key] = img
    plt.close(fig)


def histogram(likelihoods: list[float], key: str, bins: int = 50):
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, range=(0, 1), color="#4c72b0", edgecolor="black", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count")
    ax.set_title(f"{key} (hist)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - hist", fig)


def histogram_log(likelihoods: list[float], key: str, bins: int = 50):
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, range=(0, 1), color="#55a868", edgecolor="black", alpha=0.8, log=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{key} (hist, log10(count))")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - hist_logy", fig)


def cdf(likelihoods: list[float], key: str):
    x = np.sort(as_np(likelihoods))
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#c44e52")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("CDF")
    ax.set_title(f"{key} (CDF)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - cdf", fig)


def violin_plot(likelihoods: list[float], key: str):
    vals = np.asarray(likelihoods, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 0.0, 1.0)

    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot(vals, showmeans=True, showmedians=True)
    ax.set_ylabel("Likelihood")
    ax.set_xticks([])
    ax.set_title(f"{key} (violin)")

    _log_image(f"{key} - violin", fig)


def ecdf_slope(likelihoods: list[float], key: str, bins: int = 80):
    # Approximate ECDF slope as a density estimate (histogram normalized).
    x = as_np(likelihoods)
    if x.size == 0:
        return
    counts, edges = np.histogram(x, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(centers, counts, color="#8172b2")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("ECDF slope proxy / density")
    ax.set_title(f"{key} (ECDF slope proxy / density)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - ecdf_slope", fig)


##################################################################################


# Main


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)
    wandb.init(project="cosi-betadecay-likelihoods")

    energy_likelihoods, klein_nishina_weight_likelihoods = annihilation_extractor_test_likelihoods(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )

    # Energy
    histogram(energy_likelihoods, "Energy Likelihood (Compton Kinematic Filter)")
    histogram_log(energy_likelihoods, "Energy Likelihood (Compton Kinematic Filter)")
    cdf(energy_likelihoods, "Energy Likelihood (Compton Kinematic Filter)")
    violin_plot(energy_likelihoods, "Energy Likelihood (Compton Kinematic Filter)")
    ecdf_slope(energy_likelihoods, "Energy Likelihood (Compton Kinematic Filter)")

    # Klein Nishina
    histogram(klein_nishina_weight_likelihoods, "Klein Nishina Weight")
    histogram_log(klein_nishina_weight_likelihoods, "Klein Nishina Weight")
    cdf(klein_nishina_weight_likelihoods, "Klein Nishina Weight")
    violin_plot(klein_nishina_weight_likelihoods, "Klein Nishina Weight")
    ecdf_slope(klein_nishina_weight_likelihoods, "Klein Nishina Weight")

    wandb.finish()
