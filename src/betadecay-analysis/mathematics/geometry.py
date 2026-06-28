import torch


def theta_geo(
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
) -> torch.Tensor:
    """Geometric Compton scatter angle at the first interaction, in radians.

    The angle between the reconstructed incoming photon direction
    (``reconstructed_unit_vector``) and the first scatter leg (first hit ->
    second hit), per subset.

    Args:
        positions: ``(B, n, 3)`` hit positions (or ``(n, 3)`` for one subset).
        reconstructed_unit_vector: Incoming source direction, ``(3,)`` or ``(B, 3)``.

    Returns:
        Scatter angle in radians, shape ``(B,)``.
    """
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
    """Kinematic Compton scatter angle, in radians.

    Standard Compton kinematics: the incoming photon energy ``E_1`` is the total
    measured deposit and ``E_2`` is the energy remaining after the first hit.
    Disagreement with :func:`theta_geo` is the angular residual that ARM scores.

    The 511 keV constant below is the electron rest-mass energy ``m_e c²`` that
    appears in the Compton formula itself — *not* a hypothesis on the incoming
    photon energy. The 511-keV β⁺ consistency test lives in the energy ``push``
    term of :func:`physics.physics_factors.arm_fixed_size`, not here.

    Args:
        energies: ``(B, n)`` per-hit energies (or ``(n,)`` for one subset).

    Returns:
        Scatter angle in radians, shape ``(B,)``.
    """
    electron_mass_energy = 511.0  # keV (m_e c², the Compton-formula constant)
    E_1 = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    E_2 = energies[1:].sum() if energies.ndim == 1 else energies[:, 1:].sum(dim=1)

    cos_theta_kin = 1 - electron_mass_energy * (1 / E_2 - 1 / E_1)
    cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
    return torch.arccos(cos_theta_kin)
