import torch


def annihilation_kernel(event_1_positions: torch.Tensor, event_2_positions: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        event_1_positions (torch.Tensor): _description_
        event_2_positions (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    pos_1 = event_1_positions if event_1_positions.ndim == 3 else event_1_positions.unsqueeze(0)
    pos_2 = event_2_positions if event_2_positions.ndim == 3 else event_2_positions.unsqueeze(0)

    x0 = pos_1[:, 0, :]  # (B, 3)
    x1 = pos_1[:, 1, :]  # (B, 3)
    x0, x1 = x0[0], x1[0]

    y0 = pos_2[:, 0, :]  # (B, 3)
    y1 = pos_2[:, 1, :]  # (B, 3)
    y0, y1 = y0[0], y1[0]

    u_1 = (x1 - x0) / torch.clamp(torch.norm((x1 - x0), min=1e-3))
    u_2 = (y1 - y0) / torch.clamp(torch.norm((y1 - y0), min=1e-3))

    cos_alpha = torch.dot(u_1, u_2)
    mu = -1
    sigma = 0.523598776  # 30°

    return torch.exp(-((cos_alpha - mu) ** 2) / (2 * sigma**2))
