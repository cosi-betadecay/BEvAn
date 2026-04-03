from itertools import combinations

import torch


def annihilation_kernel(positions: torch.Tensor) -> torch.Tensor:
    def pca_direction(points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] < 2:
            return torch.zeros(3, device=points.device, dtype=points.dtype)

        centered = points - points.mean(dim=0, keepdim=True)
        cov = centered.T @ centered

        # Robust eigh path: run in float64, then cast back
        cov64 = cov.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(cov64)
        eigvals = eigvals.to(points.dtype)
        eigvecs = eigvecs.to(points.dtype)

        direction = eigvecs[:, torch.argmax(eigvals)]
        return direction / torch.clamp(torch.norm(direction), min=1e-6)

    def branch_heuristic(points: torch.Tensor):
        n = points.shape[0]
        idx = list(range(n))
        best_cos_alpha = 1.0

        # Enumerate non-trivial 2-way splits; fix index 0 in A to remove symmetric duplicates.
        for r in range(2, n - 1):
            for comb in combinations(idx[1:], r - 1):
                a_idx = {0, *comb}
                b_idx = [i for i in idx if i not in a_idx]
                a_idx = sorted(a_idx)
                if len(a_idx) < 2 or len(b_idx) < 2:
                    continue

                a = points[a_idx]
                b = points[b_idx]
                u1 = pca_direction(a)
                u2 = pca_direction(b)
                cos_alpha_temp = torch.dot(u1, u2).item()

                # Keep the most back-to-back split (cos closest to -1).
                if cos_alpha_temp < best_cos_alpha:
                    best_cos_alpha = cos_alpha_temp

        return best_cos_alpha

    points = positions.reshape(-1, 3)
    points = points[torch.any(points != 0, dim=1)]
    points = torch.unique(points, dim=0)
    if points.shape[0] < 4:
        return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

    cos_val = branch_heuristic(points)
    cos_val = torch.tensor(cos_val, device=positions.device, dtype=positions.dtype)

    mu, sigma = -1.0, 0.05
    scores = torch.exp(-((cos_val - mu) ** 2) / (2 * sigma**2))
    return scores
