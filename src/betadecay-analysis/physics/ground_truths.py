import ROOT as M

from utils.megalib_types import MSimEvent


def calculate_tolerance(n_std: int = 3) -> float:
    """Calculate the physically motivated energy tolerance for 511 keV detection.

    This function converts the detector's energy resolution, given as a full width
    at half maximum (FWHM), into a standard deviation sigma. For Gaussian-shaped
    peaks (as in High-Purity Germanium detectors), the relationship between FWHM
    and sigma is:

        FWHM = 2.355 * sigma

    Therefore, sigma = FWHM / 2.355.

    Args:
        n_std (int): Number of resolution sigmas defining the 511 keV window
            (default ``3``, the deployed label).

    Returns:
        float: The energy tolerance in keV (``n_std`` × σ; 3σ by default).
    """
    fwhm = 2.25
    sigma = fwhm / 2.355
    tolerance = n_std * sigma
    return tolerance


def anni_process_energies(event: MSimEvent) -> list[float]:
    """Summed secondary-hit energy (keV) of each ANNI process in the event.

    One entry per ANNI interaction: the total energy of every hit originating from
    that annihilation's secondary particles. Both the 511 keV label and its
    n_std-free score are thresholds on these per-process sums, so keeping the
    computation in one place stops the two from drifting apart.

    Args:
        event (MSimEvent): MEGAlib event object containing interactions and hits.

    Returns:
        list[float]: One summed secondary energy (keV) per ANNI process.
    """
    energies: list[float] = []
    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            process_id = i + 1
            secondary_ids = [
                j + 1 for j in range(event.GetNIAs()) if event.GetIAAt(j).GetOriginID() == process_id
            ]

            total_energy = 0
            for h in range(event.GetNHTs()):
                for sid in secondary_ids:
                    if event.GetHTAt(h).IsOrigin(sid):
                        total_energy += event.GetHTAt(h).GetEnergy()
                        break
            energies.append(total_energy)
    return energies


def ground_truth_bdecay(event: MSimEvent, n_std: int = 3) -> bool:
    """Whether any ANNI process's secondaries sum to within the 511 keV window.

    Args:
        event (MSimEvent): MEGAlib event object containing interactions and hits.
        n_std (int): Number of resolution sigmas defining the 511 keV photopeak
            window. The deployed label uses ``3``; the ``gt_tolerance`` ablation
            sweeps it (cheaply, via :func:`bdecay_label_score`) to show the study's
            conclusions do not hinge on the cut.

    Returns:
        bool: True if any ANNI process matches 511 keV within tolerance, else False.
    """
    tolerance = calculate_tolerance(n_std)
    return any(abs(total_energy - 511.0) < tolerance for total_energy in anni_process_energies(event))


def bdecay_label_score(event: MSimEvent) -> float:
    """Smallest ``|ANNI sum - 511|`` (keV) over the event's ANNI processes; inf if none.

    The n_std-independent statistic the label thresholds: an event is β⁺ for a window
    ``n_std`` iff this score ``< calculate_tolerance(n_std)``. Caching it per event lets
    the ``gt_tolerance`` sweep relabel for every window from a single extraction instead
    of re-streaming the ``.sim`` per value, while landing on exactly the labels a fresh
    extraction at that window would (the comparison is the same strict ``<``).

    Args:
        event (MSimEvent): MEGAlib event object containing interactions and hits.

    Returns:
        float: ``min(|sum - 511|)`` over ANNI processes, or ``inf`` if there are none.
    """
    return min(
        (abs(total_energy - 511.0) for total_energy in anni_process_energies(event)), default=float("inf")
    )
