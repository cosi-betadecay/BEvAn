import numpy as np
import torch
from filters import compton_kinematic_angle_filter


def energy_likelihood(energies: torch.Tensor, ref_energy: float, sigma_e: float) -> float:
    """Calculate the likelihood of a set of energy deposits given a reference energy and uncertainty.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        ref_energy (float): Reference energy (keV).
        sigma_e (float): Energy uncertainty (keV).

    Returns:
        float: Energy likelihood value.
    """
    if compton_kinematic_angle_filter(energies):
        likelihood = torch.exp(-(((torch.sum(energies) - ref_energy) / sigma_e) ** 2) / 2)
    else:
        likelihood = 0.0
    return float(likelihood)


def maximum_interaction_distance_likelihood(positions: torch.Tensor, mid_threshold: float = 4.0) -> float:
    """Calculate the maximum interaction distance likelihood for a set of interaction positions.

    Args:
        positions (torch.Tensor): Tensor of interaction positions.
        mid_threshold (float, optional): Maximum allowed distance between consecutive interaction points. Defaults to 4.0.

    Returns:
        float: Maximum interaction distance likelihood.
    """

    def sigma_d() -> float:
        """Calculate the uncertainty in interaction distance.

        Returns:
            float: Uncertainty in interaction distance.
        """
        fwhm = 0.175
        sigma = np.sqrt(2) * (fwhm / 2.355)
        return 3 * sigma

    n_interactions = positions.shape[0]
    if n_interactions < 2:
        return 1.0

    diffs = positions[1:] - positions[:-1]
    distances = torch.norm(diffs, dim=1)
    _sigma_d = sigma_d()

    if torch.isnan(distances).any():
        return 0.0

    penalties = torch.where(
        distances <= mid_threshold,
        torch.ones_like(distances),
        torch.exp(-0.5 * ((distances - mid_threshold) / _sigma_d) ** 2),
    )

    return float(torch.min(penalties))
