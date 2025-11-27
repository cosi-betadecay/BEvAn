import itertools
from typing import Any, Tuple

import ROOT as M
import wandb

from utils.reader_extraction import get_reader


def process(Event: Any, ref_energy: float, tolerance: float) -> int:
    """Count annihilation events matching a reference energy within tolerance.

    Args:
        Event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        int: Number of events that match the annihilation criteria.
    """
    NumberGoodEvents = 0

    for i in range(Event.GetNIAs()):
        if Event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            ProcessID = i + 1

            SecondaryIDs = []
            for j in range(Event.GetNIAs()):
                if Event.GetIAAt(j).GetOriginID() == ProcessID:
                    SecondaryIDs.append(j + 1)

            TotalEnergy = 0
            for h in range(Event.GetNHTs()):
                for SID in SecondaryIDs:
                    if Event.GetHTAt(h).IsOrigin(SID):
                        TotalEnergy += Event.GetHTAt(h).GetEnergy()
                        break

            if abs(TotalEnergy - ref_energy) < tolerance:
                NumberGoodEvents += 1

    return NumberGoodEvents


def detected_511_event(ref_energy: float, Event: Any, tolerance: float) -> bool:
    """Check if event contains a hit combination summing to the reference energy.

    Args:
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        Event (Any): MEGAlib event object with detector hits.
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        bool: True if a hit combination matches the reference energy, else False.
    """
    n_hits = Event.GetNHTs()

    energies = [Event.GetHTAt(i).GetEnergy() for i in range(n_hits)]

    if energies == []:
        return False

    for r in range(1, n_hits + 1):
        for combo in itertools.combinations(energies, r):
            if abs(sum(combo) - ref_energy) < tolerance:
                return True

    return False


def annihilation_extractor(
    geometry_file: str, sim_file: str, ref_energy: int = 511, tolerance: float = 1.5
) -> Tuple[int, int, int, int, float, float, float]:
    """Extract annihilation events and compute detection performance metrics.

    Args:
        geometry_file (str): Path to the detector geometry setup file (XML).
        sim_file (str): Path to the MEGAlib simulation file (.sim).
        ref_energy (float): Reference energy to compare against (default 511 keV).
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        Tuple[int, int, int, int, float, float, float]: Evaluation results as
            (TP, FP, FN, TN, precision, recall, false positive rate).
    """
    Reader = get_reader(geometry_file, sim_file)

    TP, FP, FN, TN = 0, 0, 0, 0

    while True:
        Event = Reader.GetNextEvent()
        if not Event:
            break
        M.SetOwnership(Event, True)

        NumberGoodEvents = process(Event, ref_energy, tolerance)
        is_annihilation = NumberGoodEvents > 0
        detected_511 = detected_511_event(ref_energy, Event, tolerance)

        if is_annihilation:
            if detected_511:
                TP += 1
            else:
                FN += 1
        else:
            if detected_511:
                FP += 1
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    wandb.log(
        {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "f1_score": f1_score,
        }
    )

    return TP, FP, FN, TN, precision, recall, fpr, f1_score
