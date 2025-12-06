import itertools
from typing import Any

import ROOT as M
import torch
import wandb
from mathematics.calculations import calculate_tolerance
from physics.filters import verify_compton_angle
from tqdm import tqdm
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


def detected_511_event(ref_energy: float, event: Any) -> bool:
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
    positions = torch.tensor([event.GetHTAt(i).GetPosition() for i in range(n_hits)], dtype=torch.float32)

    print(positions)
    print(energies)

    for r in range(1, n_hits + 1):
        for combo in itertools.combinations(energies, r):
            combo_tensor = torch.stack(combo)

            if torch.abs(combo_tensor.sum() - ref_energy) >= tolerance and verify_compton_angle(combo_tensor):
                return True

    return False


def annihilation_extractor(geometry_file: str, sim_file: str, ref_energy: int = 511) -> None:
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
        detected_511 = detected_511_event(ref_energy, event)

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
        }
    )
