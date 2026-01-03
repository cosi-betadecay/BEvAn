import numpy as np
import torch

from physics.likelihoods.compton_kin import compton_kin_heurestic_bdecay, compton_kin_heurestic_bg
from physics.likelihoods.energy import energy_heurestic_bg, energy_pdf_bdecay
from physics.likelihoods.mid import mid_heurestic_bdecay, mid_heurestic_bg


def posterior_bdecay(
    energies: torch.Tensor,
    positions: torch.Tensor,
    time: float,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_compton_kin: float,
    alpha_mid: float,
    alpha_arm: float,
) -> float:
    """Calculate the posterior score for beta decay given various pdfs, heurestics, etc.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        time (float): Interaction time.
        ref_energy (float): Reference energy (keV).
        tolerance (float): Energy tolerance (keV).
        alpha_energy (float): Energy PDF weight.
        alpha_compton_kin (float): Compton kinematic angle prior weight.
        alpha_mid (float): Maximum interaction distance prior weight.
        alpha_arm (float): Angular resolution measure prior weight.

    Returns:
        float: Posterior score.
    """
    energy_beta = energy_pdf_bdecay(energies, ref_energy, tolerance)
    compton_kin_beta = compton_kin_heurestic_bdecay(energies)
    mid_beta = mid_heurestic_bdecay(positions, time)
    # arm_beta = angular_resolution_measure_likelihood(energies, positions)

    log_posterior = (
        alpha_energy * np.log(energy_beta)
        + alpha_compton_kin * np.log(compton_kin_beta)
        + alpha_mid * np.log(mid_beta)
    )

    return np.exp(log_posterior)


def posterior_bg(
    energies: torch.Tensor,
    positions: torch.Tensor,
    time: float,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_compton_kin: float,
    alpha_mid: float,
    alpha_arm: float,
) -> float:
    """Calculate the posterior score for a background event given various various pdfs, heurestics, etc.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        time (float): Interaction time.
        ref_energy (float): Reference energy (keV).
        tolerance (float): Energy tolerance (keV).
        alpha_energy (float): Energy heurestic weight.
        alpha_compton_kin (float): Compton kinematic angle prior weight.
        alpha_kn (float): Klein-Nishina prior weight.
        alpha_mid (float): Maximum interaction distance prior weight.
        alpha_arm (float): Angular resolution measure prior weight.

    Returns:
        float: Posterior score.
    """
    energy_beta = energy_heurestic_bg(energies, ref_energy, tolerance)
    compton_kin_beta = compton_kin_heurestic_bg(energies)
    mid_beta = mid_heurestic_bg(positions, time)
    # arm_beta = angular_resolution_measure_likelihood(energies, positions)

    log_posterior = (
        alpha_energy * np.log(energy_beta)
        + alpha_compton_kin * np.log(compton_kin_beta)
        + alpha_mid * np.log(mid_beta)
    )

    return np.exp(log_posterior)
