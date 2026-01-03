import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ROOT as M
import torch
import wandb
from tqdm import tqdm

from mathematics.calculations import calculate_tolerance
from physics.posterior import posterior_bdecay, posterior_bg
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def ground_truth(event: Any, ref_energy: float) -> bool:
    """Count annihilation events matching a reference energy within tolerance.

    Args:
        Event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        bool: True if any event matches the annihilation criteria, else False.
    """
    tolerance = calculate_tolerance()
    number_good_events = 0

    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            process_id = i + 1

            secondary_ids = []
            for j in range(event.GetNIAs()):
                if event.GetIAAt(j).GetOriginID() == process_id:
                    secondary_ids.append(j + 1)

            total_energy = 0
            for h in range(event.GetNHTs()):
                for sid in secondary_ids:
                    if event.GetHTAt(h).IsOrigin(sid):
                        total_energy += event.GetHTAt(h).GetEnergy()
                        break

            if abs(total_energy - ref_energy) < tolerance:
                number_good_events += 1

    return number_good_events > 0


def classifier(posterior_bdecay: float, posterior_bg: float) -> bool:
    """_summary_

    Args:
        posterior_bdecay (float): _description_
        posterior_bg (float): _description_

    Returns:
        bool: _description_
    """
    return np.log(posterior_bdecay) - np.log(posterior_bg) >= 0


def detected_511_event(
    ref_energy: float,
    event: Any,
    alpha_energy: float,
    alpha_compton_kin: float,
    alpha_kn: float,
    alpha_mid: float,
    alpha_arm: float,
) -> bool:
    """Determine if an event is detected as a 511 keV annihilation event based on posterior probability.

    Args:
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        event (Any): MEGAlib event object containing interactions and hits.
        alpha_compton_kin (float): Compton kinetic energy parameter.
        alpha_kn (float): Klein-Nishina parameter.
        alpha_mid (float): Maximum interaction distance parameter.
        alpha_arm (float): Angular resolution measure parameter.
    Returns:
        bool: True if the event is detected as a 511 keV annihilation event, else False.
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

    _posterior_bdecay = -float("inf")
    _posterior_bg = -float("inf")

    for r in range(2, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            t = 0  # Find a way of getting time information from MEGAlib events

            _posterior_bdecay = max(
                _posterior_bdecay,
                posterior_bdecay(
                    energy_combo,
                    pos_combo,
                    t,
                    ref_energy,
                    tolerance,
                    alpha_energy,
                    alpha_compton_kin,
                    alpha_kn,
                    alpha_mid,
                    alpha_arm,
                ),
            )

            _posterior_bg = max(
                _posterior_bg,
                posterior_bg(
                    energy_combo,
                    pos_combo,
                    t,
                    ref_energy,
                    tolerance,
                    alpha_energy,
                    alpha_compton_kin,
                    alpha_kn,
                    alpha_mid,
                    alpha_arm,
                ),
            )

    return classifier(_posterior_bdecay, _posterior_bg)


def annihilation_extractor(
    geometry_file: str,
    sim_file: str,
    alpha_energy: float,
    alpha_compton_kin: float,
    alpha_kn: float,
    alpha_mid: float,
    alpha_arm: float,
    ref_energy: int = 511,
) -> None:
    """Extract annihilation events and compute detection performance metrics.

    Args:
        geometry_file (str): Path to the detector geometry setup file (XML).
        sim_file (str): Path to the MEGAlib simulation file (.sim).
        alpha_compton_kin (float): Compton kinetic energy parameter.
        alpha_kn (float): Klein-Nishina parameter.
        alpha_mid (float): Maximum interaction distance parameter.
        alpha_arm (float): Angular resolution measure parameter.
        ref_energy (float): Reference energy to compare against (default 511 keV).
    """
    reader = get_reader(geometry_file, sim_file)

    tp, fp, fn, tn = 0, 0, 0, 0

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        prediction = detected_511_event(
            ref_energy, event, alpha_energy, alpha_compton_kin, alpha_kn, alpha_mid, alpha_arm
        )

        if ground_truth(event, ref_energy):
            if prediction:
                tp += 1
            else:
                fn += 1
        else:
            if prediction:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fig = plot_confusion_matrix(tp, fp, fn, tn)
    wandb.log(
        {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "f1_score": f1_score,
            "confusion_matrix": wandb.Image(fig),
        }
    )
    plt.close(fig)
