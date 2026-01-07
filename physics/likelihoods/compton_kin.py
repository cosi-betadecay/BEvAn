import torch


def compton_kin_heurestic_bdecay(energies: torch.Tensor) -> float:
    def sigma_cos_phi(E_0: torch.Tensor, E: torch.Tensor, frac_sigma: float = 0.0035) -> float | None:
        """Calculate the uncertainty in the cosine of the Compton scattering angle.

        Args:
            E_0 (torch.Tensor): Sum of all energy deposits.
            E (torch.Tensor): Sum of all but the first energy deposit.
            frac_sigma (float): Fractional uncertainty. Defaults to 0.35% (due to HPGe detector resolution).

        Returns:
            float: Uncertainty in the cosine of the Compton scattering angle.
        """
        electron_mass_energy = 511.0  # keV

        sigma_E_0 = frac_sigma * E_0
        sigma_E = frac_sigma * E

        term0 = (sigma_E_0 / (E_0**2)) ** 2
        term1 = (sigma_E / (E**2)) ** 2

        return electron_mass_energy * torch.sqrt(term0 + term1)

    E0 = energies.sum()
    E = energies[1:].sum()

    m_e = 511.0
    cos_phi = 1.0 - m_e * (1.0 / E - 1.0 / E0)
    abs_cos = torch.abs(cos_phi)

    sigma = 3 * sigma_cos_phi(E0, E)
    z = (abs_cos - 1.0) / sigma

    weight = torch.where(z <= 0, torch.tensor(1.0, device=z.device), torch.exp(-0.5 * z**2))

    return weight


def compton_kin_heurestic_bg(energies: torch.Tensor) -> float:
    return
