import torch

from physics.likelihoods.arm_likelihood import angular_resolution_measure_likelihood
from physics.likelihoods.compton_kinematic_angle_likelihood import compton_kinematic_angle_likelihood
from physics.likelihoods.energy_likelihood import energy_likelihood
from physics.likelihoods.klein_nishina_likelihood import klein_nishina_likelihood
from physics.likelihoods.mid_likelihood import maximum_interaction_distance_likelihood


# Need to fix this properly so it actually becomes a posterior calculation, but for now this will do
def posterior(
    energies: torch.Tensor,
    positions: torch.Tensor,
    time: float,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
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
    compton_kin_ll = compton_kinematic_angle_likelihood(energies)
    kn_ll = klein_nishina_likelihood(energies)
    mid_ll = maximum_interaction_distance_likelihood(positions, time)
    arm_ll = angular_resolution_measure_likelihood(energies, positions)

    log_posterior = (
        alpha_energy * torch.log(energy_ll)
        + alpha_compton_kin * torch.log(compton_kin_ll)
        + alpha_kn * torch.log(kn_ll)
        + alpha_mid * torch.log(mid_ll)
        + alpha_arm * torch.log(arm_ll)
    )

    return float(torch.exp(log_posterior))
