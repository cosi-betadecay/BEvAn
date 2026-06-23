import torch


def theta_geo(
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
) -> torch.Tensor:
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
    return torch.arccos(cos_theta_geo)


def theta_kin(energies: torch.Tensor) -> torch.Tensor:
    # β⁺ hypothesis: the incoming photon is a 511 keV annihilation photon, so
    # E_1 is fixed at the electron rest mass rather than reconstructed from the
    # measured total deposit. This turns ARM into a 511-keV consistency test:
    # background photons at other energies should produce kinematic angles
    # inconsistent with the geometric reconstruction.
    electron_mass_energy = 511.0  # keV
    E_1 = electron_mass_energy
    E_2 = energies[1:].sum() if energies.ndim == 1 else energies[:, 1:].sum(dim=1)

    cos_theta_kin = 1 - electron_mass_energy * (1 / E_2 - 1 / E_1)
    cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
    return torch.arccos(cos_theta_kin)
