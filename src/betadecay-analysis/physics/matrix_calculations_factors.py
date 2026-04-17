import omegaconf
import torch
from mathematics.geometry import theta_geo, theta_kin
from tqdm import tqdm

############################################
# Annihilation Angle
############################################


def all_vector_cosines_matrix_calculation(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
        unbatched = True
    elif positions.ndim == 3:
        unbatched = False
    else:
        raise ValueError(f"Expected positions shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}")

    n = positions.shape[1]
    if n < 3:
        empty = positions.new_empty((positions.shape[0], 0))
        return empty.squeeze(0) if unbatched else empty

    deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    vectors = deltas[:, i, j, :]  # (B, M, 3), M = N choose 2
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

    cosine_matrix = vectors @ vectors.transpose(-1, -2)  # (B, M, M)
    m = vectors.shape[1]
    a, b = torch.triu_indices(m, m, offset=1, device=positions.device)
    cosines = cosine_matrix[:, a, b]  # (B, K)

    return cosines.amin().reshape(-1)


def all_vector_cosines_blockwise(
    positions: torch.Tensor,
    eps: float = 1e-12,
    combo_block_size: int = 1024,
    vec_block_size: int = 128,
) -> torch.Tensor:
    if positions.ndim != 3:
        raise ValueError(f"Expected positions shape (B, N, 3), got {tuple(positions.shape)}")

    n = positions.shape[1]
    if n < 3:
        return positions.new_empty((positions.shape[0], 0))

    total_b = positions.shape[0]
    global_best = torch.full((total_b,), 1.0, dtype=positions.dtype, device=positions.device)

    pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=positions.device)

    combo_starts = range(0, total_b, combo_block_size)
    for b0 in tqdm(combo_starts, desc="Annihilation angle blocks", unit="block", leave=False):
        b1 = min(b0 + combo_block_size, total_b)
        pos_chunk = positions[b0:b1]  # (Bc, N, 3)

        deltas = pos_chunk[:, :, None, :] - pos_chunk[:, None, :, :]  # (Bc, N, N, 3)
        vectors = deltas[:, pair_i, pair_j, :]  # (Bc, M, 3)
        vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

        bsz, m, _ = vectors.shape
        chunk_best = torch.full((bsz,), 1.0, dtype=vectors.dtype, device=vectors.device)

        for i0 in range(0, m, vec_block_size):
            i1 = min(i0 + vec_block_size, m)
            vi = vectors[:, i0:i1, :]  # (Bc, bi, 3)

            for j0 in range(i0, m, vec_block_size):
                j1 = min(j0 + vec_block_size, m)
                vj = vectors[:, j0:j1, :]  # (Bc, bj, 3)

                block = torch.matmul(vi, vj.transpose(-1, -2))  # (Bc, bi, bj)

                if i0 == j0:
                    bi = i1 - i0
                    bj = j1 - j0
                    diag = torch.eye(bi, bj, dtype=torch.bool, device=block.device).unsqueeze(0)
                    block = block.masked_fill(diag, 1.0)

                block_min = block.amin(dim=(1, 2))
                chunk_best = torch.minimum(chunk_best, block_min)

        global_best[b0:b1] = chunk_best

    return global_best.amin().reshape(-1)


def annihilation_angle(
    positions: torch.Tensor, n_hits: int | None = None, sizes: torch.Tensor | None = None
) -> torch.Tensor:
    if sizes is not None:
        if positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized annihilation_angle expects positions shape (B, N, 3) and sizes shape (B,), "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )
        if positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Positions batch dimension and sizes length must match, "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            group = sizes == size
            if size_int < 3:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            outputs.append(annihilation_angle(positions[group, :size_int, :], n_hits=size_int))

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    if n_hits is None:
        n_hits = positions.shape[0] if positions.ndim == 2 else positions.shape[1]

    if n_hits <= 14:
        cosines = all_vector_cosines_matrix_calculation(positions)
    else:
        blockwise_positions = positions.unsqueeze(0) if positions.ndim == 2 else positions
        cosines = all_vector_cosines_blockwise(blockwise_positions)

    if cosines.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)

    angle_diffs = torch.abs(cosines + 1.0)

    return cosines[torch.argmin(angle_diffs)].reshape(1)


############################################
# Agular Resolution Measure (ARM)
############################################


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:

    if sizes is not None:
        if energies.ndim != 2 or positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized arm expects energies shape (B, N), positions shape (B, N, 3), and sizes shape (B,), "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0] or positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies/positions batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            group = sizes == size
            if size_int < 3:
                outputs.append(positions.new_tensor(1.0).reshape(1))
                continue
            outputs.append(arm(energies[group, :size_int], positions[group, :size_int, :], cfg))
        return torch.stack(outputs).min().reshape(1)

    n_pos = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    if n_pos < 3:
        return positions.new_tensor(1.0)  # TODO: find a good replacement for this that actually makes sense

    arm_value = torch.min(torch.abs(theta_geo(positions, cfg)[0] - theta_kin(energies, cfg)[0]))

    return arm_value.reshape(1)


############################################
# Energy
############################################


def delta_E(
    energies: torch.Tensor, ref_energy: float = 511.0, sizes: torch.Tensor | None = None
) -> torch.Tensor:
    """Annihilation-energy residual |ΣE − ref_energy| (default ref = 511 keV).

    This is **not** Zoglauer's E_in (the photon energy before a Compton
    interaction, §4.5.3). It is an absolute distance from the expected total
    deposit for a fully absorbed β⁺ single-photon annihilation signature,
    used here as a signal-vs-background discriminator.
    """
    if sizes is not None:
        if energies.ndim != 2 or sizes.ndim != 1:
            raise ValueError(
                "Sized delta_E expects energies shape (B, N) and sizes shape (B,), "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        seq = torch.arange(energies.shape[1], device=energies.device).unsqueeze(0)
        valid = seq < sizes.unsqueeze(1)
        E = torch.where(valid, energies, torch.zeros_like(energies)).sum(dim=1)
    else:
        E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    delta_E = torch.abs(E - ref_energy)

    return delta_E.min().reshape(1)
