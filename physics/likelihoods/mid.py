import numpy as np
import torch


def mid_heurestic_bdecay(positions: torch.Tensor, time: float, mid_threshold: float = 4.0) -> float | None:
    def sigma_d() -> float:
        fwhm = 0.175
        sigma = np.sqrt(2) * (fwhm / 2.355)
        return 3 * sigma

    def mu_d(time: float) -> float:
        speed_of_light = 299792458  # meters per second
        time_seconds = time * 1e-9  # convert nanoseconds to seconds
        distance_meters = speed_of_light * time_seconds
        distance_cm = distance_meters * 100  # convert meters to centimeters
        return distance_cm

    n_interactions = positions.shape[0]
    if n_interactions < 2:
        return 1

    if positions.ndim == 3:
        diffs = positions[1:] - positions[:-1]
    else:
        diffs = positions[:, 1:, :] - positions[:, :-1, :]
    distances = torch.norm(diffs, dim=1)
    _sigma_d = sigma_d()
    _mu_d = mu_d(time)

    likelihoods = torch.exp(-0.5 * ((distances - _mu_d) / _sigma_d) ** 2)

    return float(torch.max(likelihoods))
