import torch

from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.compton_kin import compton_kin_heurestic_bdecay
from physics.likelihoods.energy import energy_kernel_bdecay
from physics.likelihoods.kn import klein_nishina_pdf
from physics.likelihoods.mid import mid_heurestic_bdecay


def posterior_bdecay(
    energy_combo: torch.Tensor,
    pos_combo: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
    alpha_compton_kin: float,
    alpha_mid: float,
    sizes: torch.Tensor,
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
        )

    # KN
    kn = one.clone()
    valid_kn = sizes > 1
    if valid_kn.any():
        kn[valid_kn] = klein_nishina_pdf(energy_combo[valid_kn])

    # Compton Kinematic Angle
    compton_kin = one.clone()
    valid_compton_kin = sizes > 1
    if valid_compton_kin.any():
        compton_kin[valid_compton_kin] = compton_kin_heurestic_bdecay(energy_combo[valid_compton_kin])

    # MID
    mid = one.clone()
    valid_mid = sizes > 1
    if valid_mid.any():
        mid[valid_mid] = mid_heurestic_bdecay(pos_combo[valid_mid], 0)  # Need a way of extracting time

    # Energy
    eng = energy_kernel_bdecay(energy_combo, ref_energy, tolerance)

    # Numerical stability
    arm = torch.clamp(arm, min=1e-12)
    kn = torch.clamp(kn, min=1e-12)
    eng = torch.clamp(eng, min=1e-12)

    score = alpha_arm * torch.log(arm) + alpha_energy * eng

    return score.max()
