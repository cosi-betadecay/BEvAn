import torch

from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_kernel_bdecay
from physics.likelihoods.kn import klein_nishina_pdf


def posterior_bdecay(
    energy_combo: torch.Tensor,
    pos_combo: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
    sizes: torch.Tensor,
    n_combo: int,
    device: torch.device,
) -> float:
    one = torch.ones(n_combo, device=device)

    # ARM
    arm = one.clone()
    valid_arm = sizes > 2
    if valid_arm.any():
        arm[valid_arm] = angular_resolution_measure_kernel(
            energy_combo[valid_arm],
            pos_combo[valid_arm],
        )

    # KN
    kn = one.clone()
    valid_kn = sizes > 1
    if valid_kn.any():
        kn[valid_kn] = klein_nishina_pdf(
            energy_combo[valid_kn]
        )

    # Energy
    eng = energy_kernel_bdecay(energy_combo, ref_energy, tolerance)

    # Numerical stability
    arm = torch.clamp(arm, min=1e-12)
    kn  = torch.clamp(kn,  min=1e-12)
    eng = torch.clamp(eng, min=1e-12)

    score = (
        alpha_arm * torch.log(arm)
        + alpha_energy * eng
        + alpha_kn * torch.log(kn)
    )

    return score.max()


# Not working now
def posterior_bg():
    return
