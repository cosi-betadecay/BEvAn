import torch

from physics.priors.arm_prior import angular_resolution_measure_prior
from physics.priors.compton_kinematic_angle_prior import compton_kinematic_angle_prior
from physics.priors.klein_nishina_prior import klein_nishina_prior
from physics.priors.mid_prior import maximum_interaction_distance_prior


def energy_likelihood(energies: torch.Tensor, ref_energy: float, sigma_e: float) -> float:
    """Calculate the likelihood of a set of energy deposits given a reference energy and uncertainty.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        ref_energy (float): Reference energy (keV).
        sigma_e (float): Energy uncertainty (keV).

    Returns:
        float: Energy likelihood value.
    """
    likelihood = torch.exp(-(((torch.sum(energies) - ref_energy) / sigma_e) ** 2) / 2)

    return float(likelihood)


def posterior(
    energies: torch.Tensor,
    positions: torch.Tensor,
    time: float,
    ref_energy: float,
    tolerance: float,
    alpha_compton_kin: float,
    alpha_kn: float,
    alpha_mid: float,
    alpha_arm: float,
) -> float:
    """Calculate the posterior probability of an event given various priors and likelihoods.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        time (float): Interaction time.
        ref_energy (float): Reference energy (keV).
        tolerance (float): Energy tolerance (keV).
        alpha_compton_kin (float): Compton kinematic angle prior weight.
        alpha_kn (float): Klein-Nishina prior weight.
        alpha_mid (float): Maximum interaction distance prior weight.
        alpha_arm (float): Angular resolution measure prior weight.

    Returns:
        float: Posterior probability.
    """
    energy_ll = energy_likelihood(energies, ref_energy, tolerance)
    compton_kin_ll = compton_kinematic_angle_prior(energies)
    kn_ll = klein_nishina_prior(energies)
    mid_ll = maximum_interaction_distance_prior(positions, time)
    arm_ll = angular_resolution_measure_prior(energies, positions)

    log_posterior = (
        torch.log(energy_ll)
        + alpha_compton_kin * torch.log(compton_kin_ll)
        + alpha_kn * torch.log(kn_ll)
        + alpha_mid * torch.log(mid_ll)
        + alpha_arm * torch.log(arm_ll)
    )

    return float(torch.exp(log_posterior))
