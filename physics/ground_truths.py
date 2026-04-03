from typing import Any

import ROOT as M

from mathematics.calculations import calculate_tolerance


def ground_truth_bdecay(event: Any, ref_energy: float) -> bool:
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


def ground_truth_annihilation(event: Any) -> bool:
    """Determine if an event contains an annihilation interaction with energy near 511 keV.

    Args:
        event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy for annihilation events (e.g., 511 keV).

    Returns:
        bool: True if an annihilation interaction is present with energy near ref_energy, else False.
    """
    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            return True

    return False


def ground_truth_compton(event: Any) -> bool:
    """Determine if an event contains a Compton scattering interaction.

    Args:
        event (Any): MEGAlib event object containing interactions and hits.

    Returns:
        bool: True if a Compton scattering interaction is present, else False.
    """
    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("COMP"):
            return True

    return False
