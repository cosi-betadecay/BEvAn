import omegaconf
import torch
from mathematics.geometry import theta_geo, theta_kin
from tqdm import tqdm

############################################
# Annihilation Angle
############################################

# Two-photon joint constraint (theory §7): a real β⁺ annihilation emits two
# ~511 keV photons that together deposit ~1022 keV. ANNI_SIGMA_E sets the
# energy scale (keV) over which a subset's total deposited energy is judged
# consistent with a full annihilation. ANNI_LAMBDA is the strength of a soft
# Gaussian penalty (in cosine units) added to each subset's back-to-back
# cosine: subsets whose ΣE is far from 1022 keV are nudged UP (less
# back-to-back / more background-like) while on-energy subsets are essentially
# untouched. Unlike a hard gate (which collapses recall by dropping subsets),
# every subset stays eligible — only its ranking shifts.
ANNI_TOTAL_KEV = 1022.0
ANNI_SIGMA_E = 80.0
ANNI_LAMBDA = 0.3


def _per_subset_min_cosines(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-subset most-back-to-back cosine.

    Mirrors ``all_vector_cosines_matrix_calculation`` but returns one value per
    subset (shape ``(B,)``) instead of collapsing the whole batch with a global
    ``amin``. Taking ``amin`` over the returned vector reproduces the global
    minimum exactly, so this is a behavior-preserving refactor at zero penalty;
    keeping the values per-subset is what lets the energy penalty be applied to
    each candidate before the minimum is taken.
    """
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    b, n = positions.shape[0], positions.shape[1]
    if n < 3:
        return positions.new_full((b,), float("nan"))

    deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    vectors = deltas[:, i, j, :]  # (B, M, 3), M = N choose 2
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

    cosine_matrix = vectors @ vectors.transpose(-1, -2)  # (B, M, M)
    m = vectors.shape[1]
    a, c = torch.triu_indices(m, m, offset=1, device=positions.device)
    cosines = cosine_matrix[:, a, c]  # (B, K)

    return cosines.amin(dim=1)  # (B,)


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
    positions: torch.Tensor,
    n_hits: int | None = None,
    sizes: torch.Tensor | None = None,
    energies: torch.Tensor | None = None,
) -> torch.Tensor:
    # ``energies`` (per-hit deposited energy, shape matching ``positions``:
    # (B, N) for the sized path) is threaded through so the two-photon joint
    # angle/energy constraint can gate candidate pairs. It is currently a
    # no-op pass-through (Phase 0); the returned cosine is unchanged.
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
            group_positions = positions[group, :size_int, :]
            if energies is not None:
                # Soft Gaussian energy penalty (seed #2): keep EVERY subset
                # eligible (no recall cliff like the hard gate of iter 1) but
                # add a per-subset penalty to its back-to-back cosine. A subset
                # whose total deposited energy ΣE is consistent with a full
                # ~1022 keV annihilation pays ~0; one far off-energy pays up to
                # ANNI_LAMBDA, nudging it UP (less back-to-back / more
                # background-like) so a random Compton coincidence that merely
                # looks back-to-back is less likely to win the per-event min.
                group_energies = energies[group, :size_int]
                total_E = group_energies.sum(dim=1)  # (Bg,)
                cosines = _per_subset_min_cosines(group_positions)  # (Bg,)
                penalty = ANNI_LAMBDA * (
                    1.0 - torch.exp(-((total_E - ANNI_TOTAL_KEV) ** 2) / (2.0 * ANNI_SIGMA_E ** 2))
                )
                # Penalised cosine stays in the calibrated [-1, 1] range
                # (clamp guards the rare off-energy subset already near +1).
                penalised = (cosines + penalty).clamp(max=1.0)
                finite_group = penalised[torch.isfinite(penalised)]
                if finite_group.numel() == 0:
                    outputs.append(positions.new_tensor(float("nan")).reshape(1))
                    continue
                outputs.append(finite_group.amin().reshape(1))
                continue
            outputs.append(annihilation_angle(group_positions, n_hits=size_int))

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
# Angular Resolution Measure (ARM)
############################################


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    reconstructed_unit_vector,
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
            if size_int < 2:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            outputs.append(
                arm(
                    energies[group, :size_int],
                    positions[group, :size_int, :],
                    cfg,
                    reconstructed_unit_vector,
                )
            )

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    n_pos = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    if n_pos < 2:
        return positions.new_tensor(float("nan")).reshape(1)

    arm_value = torch.min(
        torch.abs(theta_geo(positions, cfg, reconstructed_unit_vector)[0] - theta_kin(energies, cfg)[0])
    )

    return arm_value.reshape(1)


############################################
# Energy
############################################


def delta_E(
    energies: torch.Tensor, ref_energy: float = 511.0, sizes: torch.Tensor | None = None
) -> torch.Tensor:
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
