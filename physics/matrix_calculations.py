import omegaconf
import torch

from mathematics.geometry import theta_geo, theta_kin


def annihilation_angle(positions: torch.Tensor, eps: float = 1e-12, chunk_size: int = 1024) -> torch.Tensor:
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    elif positions.ndim != 3:
        raise ValueError(f"Expected positions shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}")

    n = positions.shape[1]
    if n < 3:
        return positions.new_tensor(float("nan")).reshape(1)

    deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    vectors = deltas[:, i, j, :]  # (B, M, 3)
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

    flat_vectors = vectors.reshape(-1, 3)  # (B*M, 3)
    m = flat_vectors.shape[0]

    if m < 2:
        return positions.new_tensor(float("nan")).reshape(1)

    best_cosine = positions.new_tensor(1.0)

    for start_i in range(0, m, chunk_size):
        end_i = min(start_i + chunk_size, m)
        vi = flat_vectors[start_i:end_i]  # (Ci, 3)

        for start_j in range(start_i, m, chunk_size):
            end_j = min(start_j + chunk_size, m)
            vj = flat_vectors[start_j:end_j]  # (Cj, 3)

            block = vi @ vj.T  # (Ci, Cj)

            if start_i == start_j:
                diag_mask = torch.eye(block.shape[0], dtype=torch.bool, device=block.device)
                block = block.masked_fill(diag_mask, 1.0)

            block_min = block.min()
            best_cosine = torch.minimum(best_cosine, block_min)

    return best_cosine.reshape(1)


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

    return delta_E


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
