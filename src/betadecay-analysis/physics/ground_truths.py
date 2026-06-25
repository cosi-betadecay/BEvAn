import ROOT as M

from utils.megalib_types import MSimEvent

# Detector energy resolution at the 511 keV line, as a full width at half maximum (keV).
# Single source for both the ground-truth tolerance and the 511-anchored geometry
# Gaussian, so the two can never drift apart.
ANNIHILATION_FWHM_KEV = 2.25


def calculate_tolerance(n_std: int = 3) -> float:
    """Energy tolerance for 511 keV detection: ``n_std`` standard deviations.

    Converts the detector resolution (FWHM, :data:`ANNIHILATION_FWHM_KEV`) to a Gaussian
    sigma via ``FWHM = 2.355 * sigma`` (the FWHM/sigma relation for the Gaussian photopeak
    of a high-purity germanium detector) and scales it by ``n_std``. Pass ``n_std=1`` to
    get the bare sigma — e.g. as the width of a 511 keV Gaussian.

    Args:
        n_std: Number of standard deviations the tolerance spans (default 3).

    Returns:
        float: The energy tolerance in keV (``n_std * sigma``).
    """
    sigma = ANNIHILATION_FWHM_KEV / 2.355
    return n_std * sigma


def ground_truth_bdecay(event: MSimEvent) -> bool:
    """Count annihilation events matching 511 keV within a resolution-based tolerance.

    Args:
        event (MSimEvent): MEGAlib event object containing interactions and hits.

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

            if abs(total_energy - 511.0) < tolerance:
                number_good_events += 1

    return number_good_events > 0
