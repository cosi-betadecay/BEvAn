import torch


def compton_kin(energies: torch.Tensor) -> bool:
    """Calculate the Compton scattering angle from two energy deposits.

    Args:
        energies (list): List of energy values for a Compton sequence.

    Returns:
        bool: True if the angle is physically valid, False otherwise.
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

    energies = energies[energies > 0]

    if energies.numel() < 2:
        return True

    E_0 = energies.sum()
    E = energies[1:].sum()

    electron_mass_energy = 511.0  # keV
    cos_phi = 1 - electron_mass_energy * (1 / E - 1 / E_0)

    _sigma_cos_phi = 75 * sigma_cos_phi(E_0, E)

    limit = 1.0 + _sigma_cos_phi

    return bool(torch.abs(cos_phi) <= limit)


def compton_kin_preprocessing(
    energies: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = torch.tensor(
        [compton_kin(energies[i]) for i in range(energies.shape[0])],
        device=energies.device,
        dtype=torch.bool,
    )
    valid_energies = energies[valid_mask]
    valid_positions = positions[valid_mask]

    return valid_energies, valid_positions
