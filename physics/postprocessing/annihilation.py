import torch


def annihilation_kernel(event_1_positions: torch.Tensor, event_2_positions: torch.Tensor) -> torch.Tensor:
    pos_1 = event_1_positions if event_1_positions.ndim == 3 else event_1_positions.unsqueeze(0)
    pos_2 = event_2_positions if event_2_positions.ndim == 3 else event_2_positions.unsqueeze(0)

    x0 = pos_1[:, 0, :]  # (B, 3)
    x1 = pos_1[:, 1, :]  # (B, 3)
    x0, x1 = x0[0], x1[0]

    y0 = pos_2[:, 0, :]  # (B, 3)
    y1 = pos_2[:, 1, :]  # (B, 3)
    y0, y1 = y0[0], y1[0]

    dx = x1 - x0
    dy = y1 - y0

    u_1 = dx / torch.clamp(torch.norm(dx), min=1e-3)
    u_2 = dy / torch.clamp(torch.norm(dy), min=1e-3)

    cos_alpha = torch.dot(u_1, u_2)

    mu = -1.0
    sigma = 0.05

    return torch.exp(-((cos_alpha - mu) ** 2) / (2 * sigma**2))
