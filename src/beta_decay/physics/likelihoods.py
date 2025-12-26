import torch


def energy_likelihood(energies: torch.Tensor, ref_energy: float, sigma_e: float) -> float:
    """Calculate the likelihood of a set of energy deposits given a reference energy and uncertainty.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        ref_energy (float): Reference energy (keV).
        sigma_e (float): Energy uncertainty (keV).

    Returns:
        float: Energy likelihood value.
    """
    return (
        1 / (torch.sqrt(2 * torch.pi) * sigma_e) * torch.exp(-(((energies - ref_energy) / sigma_e) ** 2) / 2)
    )


def compton_kinematic_likelihood(energies: torch.Tensor) -> float:
    """Calculate the Compton kinematic likelihood for a set of energy deposits.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
    Returns:
        float: Compton kinematic likelihood.
    """

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

    E_0 = energies.sum()
    E = energies[1:].sum()

    electron_mass_energy = 511.0  # keV

    cos_phi = 1 - electron_mass_energy * (1 / E - 1 / E_0)
    _sigma_cos_phi = sigma_cos_phi(E_0, E)

    return torch.exp(-((torch.max(torch.tensor(0.0), torch.abs(cos_phi) - 1) / _sigma_cos_phi) ** 2) / 2)
