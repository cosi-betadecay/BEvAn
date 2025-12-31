import numpy as np
import torch


def energy_likelihood(
    energies: torch.Tensor, positions: torch.Tensor, ref_energy: float, sigma_e: float
) -> float:
    """Calculate the likelihood of a set of energy deposits given a reference energy and uncertainty.

    Args:
        energies (torch.Tensor): Energy deposits (keV) in interaction order.
        positions (torch.Tensor): Interaction positions.
        ref_energy (float): Reference energy (keV).
        sigma_e (float): Energy uncertainty (keV).

    Returns:
        float: Energy likelihood value.
    """
    likelihood = torch.exp(-(((torch.sum(energies) - ref_energy) / sigma_e) ** 2) / 2)

    return float(likelihood)


# Didn't really make a difference this weight
def klein_nishina_weight(energies: torch.Tensor) -> float:
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


def maximum_interaction_distance_weight(
    positions: torch.Tensor, time: float, mid_threshold: float = 4.0
) -> float | None:
    """Calculate the maximum interaction distance likelihood for a set of interaction positions.

    Args:
        positions (torch.Tensor): Tensor of interaction positions.
        mid_threshold (float, optional): Maximum allowed distance between consecutive interaction points. Defaults to 4.0.

    Returns:
        float | None: Maximum interaction distance likelihood. None if not enough interactions.
    """

    def sigma_d() -> float:
        """Calculate the uncertainty in interaction distance.

        Returns:
            float: Uncertainty in interaction distance.
        """
        fwhm = 0.175
        sigma = np.sqrt(2) * (fwhm / 2.355)
        return 3 * sigma

    def mu_d(time: float) -> float:
        """Calculate the mean interaction distance based on time.

        Args:
            time (float): Time in nanoseconds.

        Returns:
            float: Mean interaction distance.
        """
        speed_of_light = 299792458  # meters per second
        time_seconds = time * 1e-9  # convert nanoseconds to seconds
        distance_meters = speed_of_light * time_seconds
        distance_cm = distance_meters * 100  # convert meters to centimeters
        return distance_cm

    n_interactions = positions.shape[0]
    if n_interactions < 2:
        return None

    diffs = positions[1:] - positions[:-1]
    distances = torch.norm(diffs, dim=1)
    _sigma_d = sigma_d()
    _mu_d = mu_d(time)

    likelihoods = torch.exp(-0.5 * ((distances - _mu_d) / _sigma_d) ** 2)

    return float(torch.max(likelihoods))


def compton_kinematic_angle_weight(energies: torch.Tensor) -> bool:
    """Calculate the Compton scattering angle from two energy deposits.

    Args:
        energies (list): List of energy values for a Compton sequence.

    Returns:
        bool: True if the angle is physically valid, False otherwise.
    """
    # Get's integrated to the beta decay function, that's why we return True for less than 2 hits.
    if energies.numel() < 2:
        return True

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

    limit = 1.0 + _sigma_cos_phi

    weight = torch.abs(cos_phi) / limit
    weight = weight.clamp(weight, 0, 1.0)

    return float(weight)


def probability_density_function(
    energies: torch.Tensor, positions: torch.Tensor, time: float, ref_energy: float, tolerance: float
) -> float:
    _energy_likelihood = energy_likelihood(energies, ref_energy, tolerance)
    _compton_kinematic_weight = compton_kinematic_angle_weight(energies)
    _maximum_interaction_distance_weight = maximum_interaction_distance_weight(positions, time)

    return _energy_likelihood * _compton_kinematic_weight * _maximum_interaction_distance_weight
