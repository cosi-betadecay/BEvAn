import itertools
from typing import Any

import matplotlib.pyplot as plt
import ROOT as M
import torch
import wandb
from mathematics.calculations import calculate_tolerance
from physics.filters import (
    angular_resolution_measure_filter,
    maximum_interaction_distance_filter,
    verify_compton_angle,
)
from tqdm import tqdm
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def process(event: Any, ref_energy: float) -> int:
    """Count annihilation events matching a reference energy within tolerance.

    Args:
        Event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        int: Number of events that match the annihilation criteria.
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

    return number_good_events


def detected_511_event(
    ref_energy: float,
    event: Any,
    baseline_activation: bool,
    compton_angle_activation: bool,
    mid_activation: bool,
    arm_activation: bool,
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

            if torch.abs(energy_combo.sum() - ref_energy) >= tolerance and baseline_activation:
                continue

            if not verify_compton_angle(energy_combo) and compton_angle_activation:
                continue

            if not maximum_interaction_distance_filter(pos_combo) and mid_activation:
                continue

            if len(idx_combo) >= 3 and arm_activation:
                arm_filter = angular_resolution_measure_filter(energy_combo, pos_combo)
                if arm_filter is False or arm_filter is None:
                    continue

            return True

    return False


def annihilation_extractor(
    geometry_file: str,
    sim_file: str,
    baseline_activation: bool,
    compton_angle_activation: bool,
    mid_activation: bool,
    arm_activation: bool,
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

        number_good_events = process(event, ref_energy)
        is_annihilation = number_good_events > 0
        detected_511 = detected_511_event(
            ref_energy,
            event,
            baseline_activation,
            compton_angle_activation,
            mid_activation,
            arm_activation,
        )

        if is_annihilation:
            if detected_511:
                tp += 1
            else:
                fn += 1
        else:
            if detected_511:
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
