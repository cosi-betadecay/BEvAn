import torch


def maximum_interaction_distance_filter(positions: torch.Tensor, mid_threshold: float = 4.0) -> bool:
    """Reject events where consecutive interaction points occur too far together.

    Args:
        positions (torch.Tensor):
            Tensor of shape (N, 3) containing interaction positions.
            Requires at least 2 positions; otherwise the event automatically passes.
        mid_threshold (float):
            Maximum allowed distance between consecutive interaction points.
            Defaults to 4.0

    Returns:
        bool: True if all consecutive interaction distances are <= mid_threshold. False if any distance is below threshold or if NaNs are encountered.
    """
    if positions.shape[0] < 2:
        return True

    diffs = positions[1:] - positions[:-1]  # shape (N-1, 3)
    distances = torch.norm(diffs, dim=1)  # shape (N-1,)

    if torch.isnan(distances).any():
        return False

    return bool(torch.all(distances <= mid_threshold))


def mid_preprocessing(energies: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return
