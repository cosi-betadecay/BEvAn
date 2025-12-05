import torch


def verify_compton_angle(energies: torch.Tensor) -> bool:
    """Calculate the Compton scattering angle from two energy deposits.

    Args:
        energies (list): List of energy values for a Compton sequence.

    Returns:
        bool: True if the angle is physically valid, False otherwise.
    """
    # Get's integrated to the beta decay function, that's why we return True for less than 2 hits.
    if energies.numel() < 2:
        return True

    def sigma_cos_phi(
        E_0: torch.Tensor, E: torch.Tensor, frac_sigma: float = 0.0035
    ) -> float | None:
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

    limit = 1.0 + _sigma_cos_phi

    return bool(torch.abs(cos_phi) <= limit)


def klein_nishina_filter(energies: torch.Tensor, polarization: bool) -> bool:
    def theta(E_0: float, E: float, electron_mass_energy: float) -> float:
        return

    def phi() -> float:
        return

    def unpolarized_klein_nishina(E_0: float, E: float) -> float:
        theta_val = theta(E_0, E, electron_mass_energy)

    def polarized_klein_nishina(E_0: float, E: float) -> float:
        theta_val = theta(E_0, E, electron_mass_energy)
        phi_val = phi()
        return

    r_e = 2.8179403227e-15  # classical electron radius in meters
    electron_mass_energy = 511.0  # keV
    E_0 = energies.sum()
    E = energies[1:].sum()

    if polarization:
        klein_nishina_eq = polarized_klein_nishina(E_0, E)
    else:
        klein_nishina_eq = unpolarized_klein_nishina(E_0, E)

    return


# Don't use per now, doesn't work well
def verify_energies_sequence(energies: torch.Tensor) -> bool:
    """Verify that the energies follow the expected sequence for Compton scattering.

    Args:
        energies (torch.Tensor): Tensor of energy deposits.

    Returns:
        bool: True if the energies follow the expected sequence, False otherwise.
    """
    if energies.numel() < 2:
        return True

    return energies[0] >= energies[1]
