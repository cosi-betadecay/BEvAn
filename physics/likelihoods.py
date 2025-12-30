import numpy as np
import torch

from physics.filters import compton_kinematic_angle_filter


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


# Didn't really make a difference this weight
def klein_nishina_weight(energies: torch.Tensor) -> float:
    if energies.numel() < 2:
        return 1.0

    me = 511.0
    E0 = energies.sum()
    E = energies[1:].sum()

    # Guard against nonsense combos
    if E <= 0 or E0 <= 0 or E >= E0:
        return 0.0

    cos_theta = 1.0 - me * (1.0 / E - 1.0 / E0)

    # If you want to keep your "validity with tolerance", do it elsewhere.
    # Here: enforce physical domain strictly.
    if cos_theta < -1.0 or cos_theta > 1.0:
        return 0.0

    # Numerical safety for acos/sin
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    lam = E / E0  # E'/E0

    # Dimensionless normalized KN (max = 1 at theta=0)
    w = 0.5 * (lam**2) * (lam + (1.0 / lam) - torch.sin(theta) ** 2)

    # Clamp for safety (should already be within [0,1] for valid Compton kinematics)
    w = torch.clamp(w, 0.0, 1.0)

    return float(w)


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
