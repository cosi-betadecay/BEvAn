import omegaconf
import torch
from tqdm import tqdm

from mathematics.geometry import theta_geo, theta_kin


def annihilation_angle(positions: torch.Tensor, n_hits: int) -> torch.Tensor:
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

    if n_hits <= 14:
        cosines = all_vector_cosines_matrix_calculation(positions)
    else:
        cosines = all_vector_cosines_blockwise(positions)

    if cosines.numel() == 0:
        return positions.new_tensor(float("nan"))

    angle_diffs = torch.abs(cosines + 1.0)

    return cosines[torch.argmin(angle_diffs)].reshape(1)


def arm(
    energies: torch.Tensor, positions: torch.Tensor, cfg: omegaconf.dictconfig.DictConfig
) -> torch.Tensor:
    if positions.shape[1] < 3:
        return 1  # TODO: find a good replacement for this that actually makes sense

    arm = torch.min(torch.abs(theta_geo(positions, cfg)[0] - theta_kin(energies, cfg)[0]))

    return arm


def delta_E(energies: torch.Tensor, ref_energy: float = 511.0) -> torch.Tensor:
    E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    delta_E = torch.abs(E - ref_energy)

    return delta_E.min().reshape(1)


def build_density_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    n_bins_x: int = 50,
    n_bins_y: int = 50,
) -> torch.Tensor:
    x = x.flatten()
    y = y.flatten()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Cannot create probability matrix from empty tensors.")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if x_min == x_max:
        x_max = x_max + 1e-6
    if y_min == y_max:
        y_max = y_max + 1e-6

    x_bins = torch.linspace(x_min, x_max, n_bins_x + 1, device=x.device, dtype=x.dtype)
    y_bins = torch.linspace(y_min, y_max, n_bins_y + 1, device=y.device, dtype=y.dtype)

    x_idx = torch.bucketize(x, x_bins) - 1
    y_idx = torch.bucketize(y, y_bins) - 1

    valid_idx = (x_idx >= 0) & (x_idx < n_bins_x) & (y_idx >= 0) & (y_idx < n_bins_y)

    x_idx = x_idx[valid_idx]
    y_idx = y_idx[valid_idx]

    counts = torch.zeros((n_bins_x, n_bins_y), dtype=torch.float32, device=x.device)
    counts.index_put_(
        (x_idx, y_idx),
        torch.ones_like(x_idx, dtype=torch.float32),
        accumulate=True,
    )

    probs = counts / counts.sum().clamp_min(1e-8)

    return probs
