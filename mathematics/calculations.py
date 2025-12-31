

def calculate_tolerance() -> float:
    """Calculate the physically motivated energy tolerance for 511 keV detection.

    This function converts the detector's energy resolution, given as a full width
    at half maximum (FWHM), into a standard deviation sigma. For Gaussian-shaped
    peaks (as in High-Purity Germanium detectors), the relationship between FWHM
    and sigma is:

        FWHM = 2.355 * sigma

    Therefore, sigma = FWHM / 2.355.

    A ±3σ window is used as the tolerance because ~99.7% of all measurements from
    a true 511 keV photon fall within this range. This provides a physically
    justified and statistically robust threshold for classifying annihilation
    events without excessively increasing false positives.

    Returns:
        float: The energy tolerance in keV (3σ).
    """
    fwhm = 2.25
    sigma = fwhm / 2.355
    tolerance = 3 * sigma
    return tolerance
