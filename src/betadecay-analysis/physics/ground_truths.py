import ROOT as M
from utils.megalib_types import MSimEvent


def ground_truth_bdecay(event: MSimEvent) -> bool:
    """Count annihilation events matching 511 keV within a resolution-based tolerance.

    Args:
        event (MSimEvent): MEGAlib event object containing interactions and hits.

    Returns:
        bool: True if any event matches the annihilation criteria, else False.
    """

    def calculate_tolerance(n_std: int = 3) -> float:
        """Calculate the physically motivated energy tolerance for 511 keV detection.

        This function converts the detector's energy resolution, given as a full width
        at half maximum (FWHM), into a standard deviation sigma. For Gaussian-shaped
        peaks (as in High-Purity Germanium detectors), the relationship between FWHM
        and sigma is:

            FWHM = 2.355 * sigma

        Therefore, sigma = FWHM / 2.355.

        Returns:
            float: The energy tolerance in keV (1σ = default).
        """
        fwhm = 2.25
        sigma = fwhm / 2.355
        tolerance = n_std * sigma
        return tolerance

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
