import torch
from mathematics.geometry import theta_geo, theta_kin

############################################
# Annihilation Angle
############################################


def per_subset_back_to_back(
    positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
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

    arm_value = torch.min(torch.abs(theta_geo(positions, reconstructed_unit_vector) - theta_kin(energies)))

    return arm_value.reshape(1)


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    reconstructed_unit_vector: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
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
    seq = torch.arange(energies.shape[1], device=energies.device).unsqueeze(0)
    valid = seq < sizes.unsqueeze(1)
    E = torch.where(valid, energies, torch.zeros_like(energies)).sum(dim=1)
    d_E = torch.abs(E - 511.0)

    return d_E.min().reshape(1)
