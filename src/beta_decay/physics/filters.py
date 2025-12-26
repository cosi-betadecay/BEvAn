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

    return bool(torch.abs(cos_phi) <= limit)


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
        electron_mass_energy = 511.0  # keV

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
    weight = 0.5 * (r_e**2) * (lam_ratio**2) * (lam_ratio + (1 / lam_ratio) - torch.sin(theta) ** 2)

    return weight


# Not too impactful because of ~1% of events having 3 or more energies/positions
def angular_resolution_measure_filter(
    energies: torch.Tensor, positions: torch.Tensor, theta_limit: float = 1.10
) -> bool | None:
    """Filter events using the angular resolution measure (ARM).

    Args:
        energies (torch.Tensor): Energy deposits for the interaction sequence (keV).
        positions (torch.Tensor): Interaction positions; first three hits are used (same order as energies).
        theta_limit (float, optional): Max allowed ``cos(theta)`` magnitude to accept noisy angles. Defaults to 1.10.

    Returns:
        bool | None: True if the ARM is within threshold, False if rejected, None when not enough hits.
    """

    def theta_geo(positions: torch.Tensor) -> float | None:
        """Compute the geometric scatter angle from the first three interaction positions.

        Args:
            positions (torch.Tensor): Interaction positions (at least three 3D points).

        Returns:
            float | None: Scatter angle in radians or None for degenerate/invalid geometry.
        """
        x0, x1, x2 = positions[0], positions[1], positions[2]

        v_in = x1 - x0
        v_out = x2 - x1

        # Degenerate geometry -> skip event
        if torch.norm(v_in) < 1e-6 or torch.norm(v_out) < 1e-6:
            return None

        cos_theta = torch.dot(v_in, v_out) / (torch.norm(v_in) * torch.norm(v_out))

        # Physically impossible -> skip event
        if torch.abs(cos_theta) > theta_limit:
            return None

        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        return torch.arccos(cos_theta)

    def theta_kin(energies: torch.Tensor) -> float | None:
        """Compute the kinematic scatter angle from the deposited energies.

        Args:
            energies (torch.Tensor): Energy deposits (keV) in interaction order.

        Returns:
            float | None: Scatter angle in radians or None if energies are unphysical.
        """
        electron_mass_energy = 511.0  # keV
        E_0 = energies.sum()
        E = energies[1:].sum()

        # Prevent division by zero
        if E_0 <= 0 or E <= 0:
            return None

        cos_theta = 1 - electron_mass_energy * (1 / E - 1 / E_0)

        if torch.abs(cos_theta) > theta_limit:
            return None

        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        return torch.arccos(cos_theta)

    def calculate_arm(theta_geo: float, theta_kin: float) -> float:
        """Calculate the ARM as the absolute difference between geometric and kinematic angles.

        Args:
            theta_geo (float): Geometric scatter angle in radians.
            theta_kin (float): Kinematic scatter angle in radians.

        Returns:
            float: ARM value in radians.
        """
        return torch.abs(theta_geo - theta_kin)

    def classify_arm(arm_calculated: float, arm_threshold: float = 0.05) -> bool:
        """Classify an event based on its ARM value.

        Args:
            arm_calculated (float): Computed ARM value in radians.
            arm_threshold (float, optional): Acceptance threshold in radians. Defaults to 0.3 (17.19°).

        Returns:
            bool: True if the ARM passes the threshold, otherwise False.
        """
        return bool(arm_calculated <= arm_threshold)

    if energies.numel() < 3 or positions.shape[0] < 3:
        return None

    _theta_geo = theta_geo(positions)
    _theta_kin = theta_kin(energies)

    if _theta_geo is None or _theta_kin is None:
        return False

    arm = calculate_arm(_theta_geo, _theta_kin)

    return classify_arm(arm)


# need to find the unit of the positions
# need to verify the mid_threshold amount - read papers
# todo: absorbtion of
# naive bayesian for prob not filters
def maximum_interaction_distance_filter(positions: torch.Tensor, mid_threshold: float = 4.0) -> bool:
    """Reject events where consecutive interaction points occur too far together.

    Args:
        positions (torch.Tensor):
            Tensor of shape (N, 3) containing interaction positions.
            Requires at least 2 positions; otherwise the event automatically passes.
        mid_threshold (float):
            Maximum allowed distance between consecutive interaction points.
            Defaults to 4.0

    Returns:
        bool: True if all consecutive interaction distances are <= mid_threshold. False if any distance is below threshold or if NaNs are encountered.
    """
    if positions.shape[0] < 2:
        return True

    diffs = positions[1:] - positions[:-1]  # shape (N-1, 3)
    distances = torch.norm(diffs, dim=1)  # shape (N-1,)

    if torch.isnan(distances).any():
        return False

    return bool(torch.all(distances <= mid_threshold))
