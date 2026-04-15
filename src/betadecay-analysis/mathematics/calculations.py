import torch


def calculate_tolerance(n_std: int = 1) -> float:
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


def min_max_norm(arr: torch.Tensor, basis: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Apply min-max normalization to the input tensor.

    Args:
        arr (torch.Tensor): Input tensor to be normalized.
        basis (torch.Tensor | None, optional): Tensor used to compute min/max for normalization. Defaults to None.
        eps (float, optional): Small value to prevent division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    ref = arr if basis is None else basis
    return (arr - ref.min()) / (ref.max() - ref.min() + eps)
