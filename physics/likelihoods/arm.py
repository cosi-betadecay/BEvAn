import torch


def angular_resolution_measure_kernel(energies: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    def theta_geo(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_x = 4.0

        x0, x1, x2 = positions[0], positions[1], positions[2]

        v_in = x1 - x0
        v_out = x2 - x1

        L_in = torch.clamp(torch.norm(v_in), min=1e-3)
        L_out = torch.clamp(torch.norm(v_out), min=1e-3)

        cos_theta_geo = torch.dot(v_in, v_out) / (L_in * L_out)
        cos_theta_geo = torch.clamp(cos_theta_geo, -1.0, 1.0)
        theta_geo = torch.arccos(cos_theta_geo)

        sigma_theta_in = torch.sqrt(torch.tensor(2.0, device=positions.device)) * sigma_x / L_in
        sigma_theta_out = torch.sqrt(torch.tensor(2.0, device=positions.device)) * sigma_x / L_out

        _sigma_theta_geo = torch.sqrt(sigma_theta_in**2 + sigma_theta_out**2)

        return theta_geo, _sigma_theta_geo

    def theta_kin(energies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def sigma_theta_kin(
            E_0: float, E: float, electron_mass_energy: float, theta_kin: float, frac_sigma: float = 0.0035
        ) -> torch.Tensor:
            sigma_cos_theta_kin = (
                electron_mass_energy * frac_sigma * torch.sqrt((1.0 / E_0) ** 2 + (1.0 / E) ** 2)
            )
            sin_theta_kin = torch.clamp(torch.abs(torch.sin(theta_kin)), min=1e-3)

            return 3 * sigma_cos_theta_kin / sin_theta_kin

        electron_mass_energy = 511.0  # keV
        E_0 = energies.sum()
        E = energies[1:].sum()

        cos_theta_kin = 1 - electron_mass_energy * (1 / E - 1 / E_0)
        cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
        theta_kin = torch.arccos(cos_theta_kin)
        _sigma_theta_kin = sigma_theta_kin(E_0, E, electron_mass_energy, theta_kin)

        return theta_kin, _sigma_theta_kin

    def arm_gaussian_kernel(arm: torch.Tensor, var_arm: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(arm**2 / var_arm) / 2)

    _theta_geo, _sigma_theta_geo = theta_geo(positions)
    _theta_kin, _sigma_theta_kin = theta_kin(energies)

    arm = _theta_geo - _theta_kin
    var_arm = _sigma_theta_geo**2 + _sigma_theta_kin**2
    var_arm = torch.clamp(var_arm, min=1e-6)

    return arm_gaussian_kernel(arm, var_arm)


# Add for bg too
