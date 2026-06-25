import torch

from mathematics.geometry import theta_geo, theta_kin

############################################
# Annihilation Angle
############################################


def per_subset_back_to_back(
    positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Vertex-anchored back-to-back annihilation score, one value per subset.

    For each hit taken as the annihilation vertex, scores how anti-parallel the
    two most-opposed arms are (cosine ~ -1 for a real pair), gated on the
    two-arm energy clearing a single-photon floor and penalized toward the
    1022 keV two-photon target. The subset's score is the best (lowest) vertex.

    Args:
        positions: ``(B, n, 3)`` hit positions for B subsets of size n.
        energies: ``(B, n)`` per-hit deposited energy.
        eps: Norm floor when normalizing arm directions.

    Returns:
        ``(B,)`` scores; ``+inf`` where no vertex passes the gate or ``n < 3``.
    """
    B2B_E_FLOOR = 511.0 - 3.0 * (2.25 / 2.355)  # ~508 keV: a single Compton photon
    B2B_E_TARGET = 1022.0  # two fully-absorbed 511 keV photons
    B2B_SIGMA_E = 250.0  # deficit-tolerant energy scale (partial deposits dominate)
    B2B_LAMBDA = 0.3  # energy-penalty strength, in cosine units

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
        deficit = (B2B_E_TARGET - e_arms).clamp_min(0.0)
        e_term = B2B_LAMBDA * (1.0 - torch.exp(-(deficit**2) / (2.0 * B2B_SIGMA_E**2)))
        score = torch.where(e_arms > B2B_E_FLOOR, cos_bb + e_term, inf)
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
    """ARM for one fixed-size subset batch: min |theta_geo - theta_kin|."""
    n_pos = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    if n_pos < 2:
        return positions.new_tensor(float("nan")).reshape(1)

    arm_value = torch.min(theta_geo(positions, reconstructed_unit_vector) - theta_kin(energies))

    return arm_value.reshape(1)


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Per-event Angular Resolution Measure, pooled over the subset sizes.

    Groups subsets by size and returns the smallest ``|theta_geo - theta_kin|``
    (via :func:`arm_fixed_size`) across all sizes ``>= 2``, or NaN when none
    qualify.

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
    return finite.amin().reshape(1)


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
