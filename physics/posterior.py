import torch

from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_pdf_bdecay
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

    arm = angular_resolution_measure_kernel(energy_combo, pos_combo)
    kn = klein_nishina_pdf(energy_combo)
    eng = energy_pdf_bdecay(energy_combo, ref_energy, tolerance)

    # Apply interaction-count logic without Python branching
    arm = torch.where(sizes > 2, arm, one)
    kn = torch.where(sizes > 1, kn, one)

    # Numerical stability
    arm = torch.clamp(arm, min=1e-12)
    kn = torch.clamp(kn, min=1e-12)
    eng = torch.clamp(eng, min=1e-12)

    score = alpha_arm * torch.log(arm) + alpha_energy * eng + alpha_kn * torch.log(kn)

    return score.max()


# Not working now
def posterior_bg():
    return
