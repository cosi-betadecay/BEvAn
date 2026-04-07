import omegaconf
import torch


def theta_geo(
    positions: torch.Tensor, cfg: omegaconf.dictconfig.DictConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_x = cfg.arm.sigma_x

    pos = positions if positions.ndim == 3 else positions.unsqueeze(0)
    x0 = pos[:, 0, :]  # (B, 3)
    x1 = pos[:, 1, :]  # (B, 3)
    x2 = pos[:, 2, :]  # (B, 3)

    v_in = x1 - x0
    v_out = x2 - x1

    L_in = torch.clamp(torch.norm(v_in, dim=1), min=1e-3)
    L_out = torch.clamp(torch.norm(v_out, dim=1), min=1e-3)

    cos_theta_geo = (v_in * v_out).sum(dim=1) / (L_in * L_out)
    cos_theta_geo = torch.clamp(cos_theta_geo, -1.0, 1.0)
    theta_geo = torch.arccos(cos_theta_geo)

    sigma_theta_in = torch.sqrt(torch.tensor(2.0, device=positions.device)) * sigma_x / L_in
    sigma_theta_out = torch.sqrt(torch.tensor(2.0, device=positions.device)) * sigma_x / L_out

    _sigma_theta_geo = torch.sqrt(sigma_theta_in**2 + sigma_theta_out**2)

    return theta_geo, _sigma_theta_geo


def theta_kin(
    energies: torch.Tensor, cfg: omegaconf.dictconfig.DictConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    def sigma_theta_kin(
        E_1: float, E_2: float, electron_mass_energy: float, theta_kin: float, frac_sigma: float = 0.0035
    ) -> torch.Tensor:
        sigma_cos_theta_kin = (
            electron_mass_energy * frac_sigma * torch.sqrt((1.0 / E_1) ** 2 + (1.0 / E_2) ** 2)
        )
        sin_theta_kin = torch.clamp(torch.abs(torch.sin(theta_kin)), min=1e-3)
        n_sigma_cos_theta_kin = cfg.arm.n_sigma_cos_theta_kin

        return n_sigma_cos_theta_kin * sigma_cos_theta_kin / sin_theta_kin

    electron_mass_energy = 511.0  # keV
    if energies.ndim == 1:
        E_1 = energies[1:].sum()
        E_2 = energies[2:].sum()
    else:
        E_1 = energies[:, 1:].sum(dim=1)
        E_2 = energies[:, 2:].sum(dim=1)

    cos_theta_kin = 1 - electron_mass_energy * (1 / E_2 - 1 / E_1)
    cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
    theta_kin = torch.arccos(cos_theta_kin)
    _sigma_theta_kin = sigma_theta_kin(E_1, E_2, electron_mass_energy, theta_kin)

    return theta_kin, _sigma_theta_kin
