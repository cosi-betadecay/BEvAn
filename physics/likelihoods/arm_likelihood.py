import torch


# Not too impactful because of ~1% of events having 3 or more energies/positions
# Need to convert to a prior somehow
def angular_resolution_measure_likelihood(
    energies: torch.Tensor, positions: torch.Tensor, theta_limit: float = 1.10
) -> float:
    """Estimate the likelihood of an event using the angular resolution measure (ARM).

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

    # Fix: return a prior value instead of a boolean
    return classify_arm(arm)
