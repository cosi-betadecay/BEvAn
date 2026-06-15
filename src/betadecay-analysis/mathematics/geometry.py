import omegaconf
import torch


def theta_geo(
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    reconstructed_unit_vector,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_x = cfg.likelihoods.arm.sigma_x

    pos = positions if positions.ndim == 3 else positions.unsqueeze(0)
    x0 = pos[:, 0, :]  # (B, 3)
    x1 = pos[:, 1, :]  # (B, 3)

    v_in = torch.as_tensor(reconstructed_unit_vector, device=positions.device, dtype=positions.dtype)
    if v_in.ndim == 1:
        v_in = v_in.expand(x0.shape[0], -1)
    v_out = x1 - x0

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
    # β⁺ hypothesis: the incoming photon is a 511 keV annihilation photon, so
    # E_1 is fixed at the electron rest mass rather than reconstructed from the
    # measured total deposit. This turns ARM into a 511-keV consistency test:
    # background photons at other energies should produce kinematic angles
    # inconsistent with the geometric reconstruction.
    def sigma_theta_kin(
        E_2: float, electron_mass_energy: float, theta_kin: float, frac_sigma: float = 0.0035
    ) -> torch.Tensor:
        sigma_cos_theta_kin = electron_mass_energy * frac_sigma / E_2
        sin_theta_kin = torch.clamp(torch.abs(torch.sin(theta_kin)), min=1e-3)
        n_sigma_cos_theta_kin = cfg.likelihoods.arm.n_sigma_cos_theta_kin

        return n_sigma_cos_theta_kin * sigma_cos_theta_kin / sin_theta_kin

    electron_mass_energy = 511.0  # keV
    E_1 = electron_mass_energy
    if energies.ndim == 1:
        E_2 = energies[1:].sum()
    else:
        E_2 = energies[:, 1:].sum(dim=1)

    cos_theta_kin = 1 - electron_mass_energy * (1 / E_2 - 1 / E_1)
    cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
    theta_kin = torch.arccos(cos_theta_kin)
    _sigma_theta_kin = sigma_theta_kin(E_2, electron_mass_energy, theta_kin)

    return theta_kin, _sigma_theta_kin
