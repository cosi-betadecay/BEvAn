import torch


def klein_nishina_likelihood(energies: torch.Tensor) -> float:
    """Calculate the Klein–Nishina weight for a given set of energy deposits.

    Args:
        energies (torch.Tensor): Tensor of energy deposits.

    Returns:
        float: Klein–Nishina weight.
    """

    def cos_theta(E_0: float, E: float, electron_mass_energy: float) -> float:
        return 1 - electron_mass_energy * (1 / E - 1 / E_0)

    def sigma_cos_theta(
        E_0: torch.Tensor, E: torch.Tensor, electron_mass_energy: float, frac_sigma: float = 0.0035
    ) -> float | None:
        """Calculate the uncertainty in the cosine of the Compton scattering angle.

        Args:
            E_0 (torch.Tensor): Sum of all energy deposits.
            E (torch.Tensor): Sum of all but the first energy deposit.
            electron_mass_energy (float): Electron rest mass energy in keV.
            frac_sigma (float): Fractional uncertainty. Defaults to 0.35% (due to HPGe detector resolution).

        Returns:
            float: Uncertainty in the cosine of the Compton scattering angle.
        """

        sigma_E_0 = frac_sigma * E_0
        sigma_E = frac_sigma * E

        term0 = (sigma_E_0 / (E_0**2)) ** 2
        term1 = (sigma_E / (E**2)) ** 2

        return electron_mass_energy * torch.sqrt(term0 + term1)

    if energies.numel() < 2:
        return 1.0  # No weighting for single hits

    r_e = 2.8179403227e-15  # classical electron radius in meters
    electron_mass_energy = 511.0  # keV
    E_0 = energies.sum()
    E = energies[1:].sum()

    _cos_theta = cos_theta(E_0, E, electron_mass_energy)
    _sigma_cos_theta = sigma_cos_theta(E_0, E, electron_mass_energy)
    limit = 1.0 + _sigma_cos_theta

    if torch.abs(_cos_theta) > limit:
        return 0.0  # Invalid angle, return zero weight

    lam_ratio = E / E_0
    theta = torch.acos(_cos_theta)

    r_e = 2.8179403227e-15  # classical electron radius, m

    # Klein–Nishina differential cross-section (unpolarized)
    kn = 0.5 * (r_e**2) * (lam_ratio**2) * (lam_ratio + (1 / lam_ratio) - torch.sin(theta) ** 2)
    kn_max = r_e**2

    weight = float(kn / kn_max)

    return weight
