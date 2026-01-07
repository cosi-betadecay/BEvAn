import numpy as np
import torch


def mid_heurestic_bdecay(positions: torch.Tensor, time: float, mid_threshold: float = 4.0) -> float | None:
    """Calculate the maximum interaction distance score for a set of interaction positions.

    Args:
        positions (torch.Tensor): Tensor of interaction positions.
        mid_threshold (float, optional): Maximum allowed distance between consecutive interaction points. Defaults to 4.0.

    Returns:
        float | None: Maximum interaction distance likelihood. None if not enough interactions.
    """

    def sigma_d() -> float:
        """Calculate the uncertainty in interaction distance.

        Returns:
            float: Uncertainty in interaction distance.
        """
        fwhm = 0.175
        sigma = np.sqrt(2) * (fwhm / 2.355)
        return 3 * sigma

    def mu_d(time: float) -> float:
        """Calculate the mean interaction distance based on time.

        Args:
            time (float): Time in nanoseconds.

        Returns:
            float: Mean interaction distance.
        """
        speed_of_light = 299792458  # meters per second
        time_seconds = time * 1e-9  # convert nanoseconds to seconds
        distance_meters = speed_of_light * time_seconds
        distance_cm = distance_meters * 100  # convert meters to centimeters
        return distance_cm

    n_interactions = positions.shape[0]
    if n_interactions < 2:
        return 1

    diffs = positions[1:] - positions[:-1]
    distances = torch.norm(diffs, dim=1)
    _sigma_d = sigma_d()
    _mu_d = mu_d(time)

    likelihoods = torch.exp(-0.5 * ((distances - _mu_d) / _sigma_d) ** 2)

    return float(torch.max(likelihoods))


def mid_heurestic_bg(positions: torch.Tensor, time: float, mid_threshold: float = 4.0) -> float | None:
    return
