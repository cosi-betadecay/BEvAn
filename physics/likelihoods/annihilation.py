import torch


def annihilation_kernel(positions: torch.Tensor) -> torch.Tensor:
    def all_vector_cosines(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
        cosines = cosine_matrix[:, a, b]  # (B, K), K = M choose 2

        return cosines.squeeze(0) if unbatched else cosines

    cosines = all_vector_cosines(positions)
    if cosines.numel() == 0:
        if positions.ndim == 2:
            return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        return torch.zeros(positions.shape[0], device=positions.device, dtype=positions.dtype)

    mu = -1
    sigma = 0.05
    kernels = torch.exp(-0.5 * ((cosines - mu) / sigma) ** 2)

    return kernels
