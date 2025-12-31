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
from physics.annihilation_detection import ground_truth
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
):
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    if event.GetGuardRingEnergy() > 0:
        return 0.0, 0.0

    if n_hits < 2:
        return 0.0, 0.0

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
    _klein_nishina_weight = 0.0

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            _energy_likelihood = max(
                _energy_likelihood, energy_likelihood(energy_combo, ref_energy, tolerance)
            )
            if r >= 2:
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

        energy_likelihood, klein_nishina_weight_likelihood = detected_511_event_likelihoods(
            ref_energy,
            event,
        )
        energy_likelihoods.append(energy_likelihood)
        klein_nishina_weight_likelihoods.append(klein_nishina_weight_likelihood)

    return energy_likelihoods, klein_nishina_weight_likelihoods, ground_truths


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


# Calculations


def log_calculations(likelihoods: list[float], key: str):
    x = as_np(likelihoods)
    valid_count = x.size

    if valid_count == 0:
        return

    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x)
    var = np.var(x)

    percentiles = np.percentile(x, [1, 5, 10, 25, 75, 90, 95, 99])
    q01, q05, q10, q25, q75, q90, q95, q99 = percentiles
    iqr = q75 - q25

    mad = np.median(np.abs(x - median))
    mean_abs_dev = np.mean(np.abs(x - mean))
    coeff_var = std / mean if mean != 0 else np.nan

    centered = x - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    skewness = m3 / (m2**1.5) if m2 > 0 else np.nan
    kurtosis_excess = m4 / (m2 * m2) - 3 if m2 > 0 else np.nan

    trimmed_mask = (x >= q05) & (x <= q95)
    trimmed_mean = np.mean(x[trimmed_mask]) if np.any(trimmed_mask) else mean

    thresholds = [0.1, 0.5, 0.9, 0.99]
    fraction_ge = {thr: float(np.mean(x >= thr)) for thr in thresholds}
    fraction_le_10 = float(np.mean(x <= 0.1))

    wandb.log(
        {
            f"{key} - mean": mean,
            f"{key} - median": median,
            f"{key} - trimmed_mean_5_95": trimmed_mean,
            f"{key} - std": std,
            f"{key} - var": var,
            f"{key} - coeff_var": coeff_var,
            f"{key} - mad": mad,
            f"{key} - mean_abs_dev": mean_abs_dev,
            f"{key} - q01": q01,
            f"{key} - q05": q05,
            f"{key} - q10": q10,
            f"{key} - q25": q25,
            f"{key} - q75": q75,
            f"{key} - q90": q90,
            f"{key} - q95": q95,
            f"{key} - q99": q99,
            f"{key} - iqr": iqr,
            f"{key} - skewness": skewness,
            f"{key} - kurtosis_excess": kurtosis_excess,
            f"{key} - frac_ge_10pct": fraction_ge[0.1],
            f"{key} - frac_ge_50pct": fraction_ge[0.5],
            f"{key} - frac_ge_90pct": fraction_ge[0.9],
            f"{key} - frac_ge_99pct": fraction_ge[0.99],
            f"{key} - frac_le_10pct": fraction_le_10,
        }
    )


##################################################################################


# Main


if __name__ == "__main__":
    load_dotenv()

    energy_likelihoods, klein_nishina_weight_likelihoods, ground_truths = (
        annihilation_extractor_test_likelihoods(
            "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
        )
    )
    energy_likelihoods_true_events = []
    energy_likelihoods_false_events = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            energy_likelihoods_true_events.append(energy_likelihoods[i])
        else:
            energy_likelihoods_false_events.append(energy_likelihoods[i])

    runs = [
        ("true_events_energy_likelihood", energy_likelihoods_true_events),
        ("false_events_energy_likelihood", energy_likelihoods_false_events),
    ]

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-likelihoods", name=label, reinit=True)

        histogram(data, "Energy True Events")
        histogram_log(data, "Energy True Events")
        cdf(data, "Energy True Events")
        violin_plot(data, "Energy True Events")
        ecdf_slope(data, "Energy True Events")
        log_calculations(data, "Energy True Events")

        wandb.finish()
