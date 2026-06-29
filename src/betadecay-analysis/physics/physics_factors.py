import math

import torch

from physics.ground_truths import calculate_tolerance

############################################
# Compton geometry primitives
############################################


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
    term of :func:`arm_fixed_size`, not here.

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


############################################
# Annihilation Angle
############################################


def per_subset_back_to_back(
    positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Vertex-anchored back-to-back annihilation score, one value per subset.

    For each hit taken as the annihilation vertex, scores how anti-parallel the
    two most-opposed arms are (cosine ~ -1 for a real pair), gated on the
    two-arm energy clearing a single-photon floor and pushed toward the
    1022 keV two-photon target. The subset's score is the best (lowest) vertex.

    The energy push mirrors the 511-aware ARM construction at the two-photon
    target: inside 1022 +- ``2 * calculate_tolerance()`` it is zero (a clean pair sits at
    pure ``cos_bb``); outside, either miss direction penalizes (excess from a
    summed-in background hit as much as a deficit), nudging the score off the
    signal rail in proportion to ``|e_arms - 1022|``. ``calculate_tolerance()`` is the
    single shared photopeak window, so this deadzone can never drift from the
    label. (The push enters as a positive penalty rather than a signed term
    because the subset score is selected by its minimum, not by ``|score|``.)

    Args:
        positions: ``(B, n, 3)`` hit positions for B subsets of size n.
        energies: ``(B, n)`` per-hit deposited energy.
        eps: Norm floor when normalizing arm directions.

    Returns:
        ``(B,)`` scores; ``+inf`` where no vertex passes the gate or ``n < 3``.
    """
    B2B_E_FLOOR = 511.0 - calculate_tolerance()  # ~508 keV: a single Compton photon
    B2B_E_TARGET = 1022.0  # two fully-absorbed 511 keV photons
    B2B_LAMBDA = 0.3  # energy-push strength, in cosine units

    B, n, _ = positions.shape
    if n < 3:
        return positions.new_full((B,), float("inf"))
    inf = positions.new_tensor(float("inf"))
    total_E = energies.sum(dim=1)  # (B,)
    best = positions.new_full((B,), float("inf"))
    for v in range(n):
        others = [k for k in range(n) if k != v]
        D = positions[:, others, :] - positions[:, v : v + 1, :]  # (B, m, 3)
        D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min(eps)
        C = D @ D.transpose(-1, -2)  # (B, m, m)
        m = len(others)
        eye = torch.eye(m, dtype=torch.bool, device=positions.device).unsqueeze(0)
        cos_bb = C.masked_fill(eye, float("inf")).amin(dim=(1, 2))  # most anti-parallel
        e_arms = total_E - energies[:, v]  # two-arm energy (vertex excluded)
        dE2 = e_arms - B2B_E_TARGET  # miss from the two-photon target
        push = (dE2.abs() - 2.0 * calculate_tolerance()).clamp(min=0.0) / 511.0
        score = torch.where(e_arms > B2B_E_FLOOR, cos_bb + B2B_LAMBDA * push, inf)
        best = torch.minimum(best, score)
    return best


def annihilation_angle(
    positions: torch.Tensor,
    energies: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Per-event back-to-back annihilation feature, pooled over the subset sizes.

    Groups subsets by size, scores each size-``>= 3`` group with
    :func:`per_subset_back_to_back`, and returns the per-event best (lowest =
    most annihilation-like), or NaN when no subset qualifies.

    Args:
        positions: ``(B, N, 3)`` padded hit positions.
        energies: ``(B, N)`` padded per-hit energies.
        sizes: ``(B,)`` true subset size per row.

    Returns:
        The single best score as a ``(1,)`` tensor (NaN if none qualify).
    """
    outputs = []
    for size in torch.unique(sizes, sorted=True):
        size_int = int(size.item())
        if size_int < 3:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        group = sizes == size
        score = per_subset_back_to_back(positions[group, :size_int, :], energies[group, :size_int])
        finite = score[torch.isfinite(score)]
        if finite.numel() == 0:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        outputs.append(finite.amin().reshape(1))

    stacked = torch.stack(outputs).flatten()
    finite = stacked[torch.isfinite(stacked)]
    if finite.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)
    return finite.amin().reshape(1)


############################################
# Angular Resolution Measure (ARM)
############################################


def arm_fixed_size(
    energies: torch.Tensor,
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
) -> torch.Tensor:
    """511-aware ARM for one fixed-size subset batch.

    Per subset, the feature fuses the angular and energy 511-consistency tests:

        a    = (theta_geo - theta_kin) / pi          # angular residual, ~0 = agree
        dE   = sum(E) - 511                           # signed energy deviation (keV)
        push = sign(dE) * max(|dE| - calculate_tolerance(), 0) / 511.0
        feature = clamp(a + push, -1, +1)

    Inside the 511 +- ``calculate_tolerance()`` window ``push`` is zero, so a true β⁺
    photon with agreeing geometry sits at ~0. Outside it, the feature is shoved
    toward +-1 in the direction of the energy miss and in proportion to it. The
    subset minimizing ``|feature|`` is selected (must satisfy *both* tests on the
    same photon) and its signed feature returned.

    Args:
        energies: ``(B, n)`` per-hit energies for one fixed-size subset batch.
        positions: ``(B, n, 3)`` hit positions for the same batch.
        reconstructed_unit_vector: Source direction passed through to ``theta_geo``.

    Returns:
        The signed feature of the best subset as a ``(1,)`` tensor (NaN if ``n < 2``).
    """
    n_pos = positions.shape[1]
    if n_pos < 2:
        return positions.new_tensor(float("nan")).reshape(1)

    a = (theta_geo(positions, reconstructed_unit_vector) - theta_kin(energies)) / math.pi
    dE = energies.sum(dim=1) - 511.0
    push = torch.sign(dE) * (dE.abs() - calculate_tolerance()).clamp(min=0.0) / 511.0
    feature = (a + push).clamp(-1.0, 1.0)

    return feature[torch.argmin(torch.abs(feature))].reshape(1)


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Per-event 511-aware ARM, pooled over the subset sizes.

    Groups subsets by size, takes each size's best subset (smallest
    ``|feature|`` via :func:`arm_fixed_size`, the energy-modulated angular
    residual in ``[-1, 1]``), and returns the signed feature of whichever of
    those is closest to zero across all sizes ``>= 2``, or NaN when none qualify.

    Args:
        energies: ``(B, N)`` padded per-hit energies.
        positions: ``(B, N, 3)`` padded hit positions.
        reconstructed_unit_vector: Source direction passed through to ``theta_geo``.
        sizes: ``(B,)`` true subset size per row.

    Returns:
        The single best ARM as a ``(1,)`` tensor (NaN if none qualify).
    """
    outputs = []
    for size in torch.unique(sizes, sorted=True):
        size_int = int(size.item())
        group = sizes == size
        if size_int < 2:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        outputs.append(
            arm_fixed_size(
                energies[group, :size_int],
                positions[group, :size_int, :],
                reconstructed_unit_vector,
            )
        )

    stacked = torch.stack(outputs).flatten()
    finite = stacked[torch.isfinite(stacked)]
    if finite.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)
    return finite[torch.argmin(torch.abs(finite))].reshape(1)


############################################
# Energy
############################################


def delta_E(
    energies: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Per-event energy feature: minimum ``|summed subset energy - 511 keV|``.

    Sums each subset's valid (non-padding) hit energies and returns the
    smallest absolute deviation from 511 keV over all subsets.

    Args:
        energies: ``(B, N)`` padded per-hit energies.
        sizes: ``(B,)`` true subset size per row (selects the valid entries).

    Returns:
        The smallest deviation as a ``(1,)`` tensor.
    """
    seq = torch.arange(energies.shape[1], device=energies.device).unsqueeze(0)
    valid = seq < sizes.unsqueeze(1)
    E = torch.where(valid, energies, torch.zeros_like(energies)).sum(dim=1)
    d_E = torch.abs(E - 511.0)

    return d_E.min().reshape(1)
