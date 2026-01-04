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
    mid_bdecay = mid_heurestic_bdecay(positions, time)
    if mid_bdecay is None:
        return None

    return 1 - mid_bdecay


def maximum_interaction_distance_filter(positions: torch.Tensor, mid_threshold: float = 4.0) -> bool:
    """Reject events where consecutive interaction points occur too far together.

    Args:
        positions (torch.Tensor):
            Tensor of shape (N, 3) containing interaction positions.
            Requires at least 2 positions; otherwise the event automatically passes.
        mid_threshold (float):
            Maximum allowed distance between consecutive interaction points.
            Defaults to 4.0

    Returns:
        bool: True if all consecutive interaction distances are <= mid_threshold. False if any distance is below threshold or if NaNs are encountered.
    """
    if positions.shape[0] < 2:
        return True

    diffs = positions[1:] - positions[:-1]  # shape (N-1, 3)
    distances = torch.norm(diffs, dim=1)  # shape (N-1,)

    if torch.isnan(distances).any():
        return False

    return bool(torch.all(distances <= mid_threshold))