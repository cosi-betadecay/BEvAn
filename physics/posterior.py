import torch

from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_heurestic_bg


def posterior_bdecay(
    energies: torch.Tensor,
    positions: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    r: int,
) -> float:
    """Calculate the posterior score for beta decay given various pdfs, heurestics, etc.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        ref_energy (float): Reference energy (keV).
        tolerance (float): Energy tolerance (keV).
        alpha_energy (float): Energy PDF weight.
        alpha_arm (float): Angular resolution measure prior weight.
        r (int): Iteration in loop.

    Returns:
        float: Posterior score.
    """
    one = torch.tensor(1.0, dtype=energies.dtype, device=energies.device)

    energy_beta = energy_heurestic_bg(energies, ref_energy, tolerance)
    arm_beta = angular_resolution_measure_kernel(energies, positions) if r > 2 else one

    posterior = alpha_energy * energy_beta + alpha_arm * torch.log(arm_beta)

    return posterior


# Not working now
def posterior_bg(
    energies: torch.Tensor,
    positions: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    r: int,
) -> float:
    """Calculate the posterior score for a background event given various various pdfs, heurestics, etc.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        time (float): Interaction time.
        ref_energy (float): Reference energy (keV).
        tolerance (float): Energy tolerance (keV).
        alpha_energy (float): Energy heurestic weight.
        alpha_arm (float): Angular resolution measure prior weight.
        r (int): Iteration in loop.

    Returns:
        float: Posterior score.
    """
    one = torch.tensor(1.0, dtype=energies.dtype, device=energies.device)

    energy_beta = energy_heurestic_bg(energies, ref_energy, tolerance)
    arm_beta = angular_resolution_measure_kernel(energies, positions) if r > 2 else one

    posterior = alpha_energy * energy_beta + alpha_arm * torch.log(arm_beta)

    return posterior
