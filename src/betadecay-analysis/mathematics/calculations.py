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
