import torch


def klein_nishina_pdf(energies: torch.Tensor) -> float:
    """Calculate the Klein–Nishina weight for a given set of energy deposits.

    Args:
        energies (torch.Tensor): Tensor of energy deposits.

    Returns:
        float: Klein–Nishina weight.
    """

    def cos_theta(E_0: float, E: float, electron_mass_energy: float) -> float:
        return 1 - electron_mass_energy * (1 / E - 1 / E_0)

    def kn_cross_section(E_0: float, r_e: float, E: float, electron_mass_energy: float) -> float:
        _cos_theta = cos_theta(E_0, E, electron_mass_energy)
        _cos_theta = torch.clamp(_cos_theta, -1.0, 1.0)

        lam_ratio = E / E_0
        theta = torch.acos(_cos_theta)

        # Klein-Nishina differential cross-section (unpolarized)
        kn = 0.5 * (r_e**2) * (lam_ratio**2) * (lam_ratio + (1 / lam_ratio) - torch.sin(theta) ** 2)

        return kn

    def sigma_scat(electron_mass_energy: float, E_0: torch.Tensor, r_e: float) -> float:
        eps = E_0 / electron_mass_energy
        one_plus_2e = 1.0 + 2.0 * eps
        ln_term = torch.log(one_plus_2e)

        sigma_kn = (
            2.0
            * torch.pi
            * r_e
            * r_e
            * (
                (1.0 + eps) / (eps * eps) * (2.0 * (1.0 + eps) / one_plus_2e - ln_term / eps)
                + 0.5 * ln_term / eps
                - (1.0 + 3.0 * eps) / (one_plus_2e * one_plus_2e)
            )
        )

        return sigma_kn

    def pdf(energies: torch.Tensor) -> float:
        electron_mass_energy = 511.0  # keV
        r_e = 2.8179403227e-15  # meters (classical electron radius)
        E_0 = energies.sum()
        E = E_0 - energies[0]

        kn_diff_cross_section = kn_cross_section(E_0, r_e, E, electron_mass_energy)
        sigma_s = sigma_scat(electron_mass_energy, E_0, r_e)
        pdf = kn_diff_cross_section / sigma_s

        return pdf

    return pdf(energies)
