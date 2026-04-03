import omegaconf
import torch

from physics.likelihoods.annihilation import annihilation_kernel
from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_kernel_bdecay


def posterior_bdecay(
    energy_combo: torch.Tensor,
    pos_combo: torch.Tensor,
    ref_energy: float,
    cfg_likelihoods: omegaconf.dictconfig.DictConfig,
    tolerance: float,
) -> float:
    device = energy_combo.device
    n_combo = energy_combo.shape[0]
    one = torch.ones(n_combo, device=device)

    sizes = (energy_combo > 0).sum(dim=1)

    # ARM
    arm = one.clone()
    valid_arm = sizes > 2
    if valid_arm.any():
        arm[valid_arm] = angular_resolution_measure_kernel(
            energy_combo[valid_arm],
            pos_combo[valid_arm],
            cfg_likelihoods,
        )

    # Annihilation
    anni = one.clone()
    valid_anni = sizes > 3
    if valid_anni.any():
        anni[valid_anni] = annihilation_kernel(pos_combo[valid_anni])

    # Energy
    eng = energy_kernel_bdecay(energy_combo, ref_energy, tolerance)

    # Numerical stability
    arm = torch.clamp(arm, min=1e-12)
    anni = torch.clamp(anni, min=1e-12)
    eng = torch.clamp(eng, min=1e-12)

    # Weights
    alpha_energy = cfg_likelihoods.posterior.alpha_energy
    alpha_arm = cfg_likelihoods.posterior.alpha_arm

    score = 2 * arm + 2 * eng + anni

    return score.max()
