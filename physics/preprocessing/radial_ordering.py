import torch


def radial_ordering(positions: torch.Tensor, eps: float = 1e-3) -> bool:
    # Keep only real hits
    mask = torch.norm(positions, dim=1) > 0
    hits = positions[mask]

    if hits.shape[0] < 2:
        return True

    r0 = hits[0]

    # Distances from first hit
    d = torch.norm(hits - r0, dim=1)

    return bool(torch.all(d[1:] >= d[:-1] - eps))


def radial_ordering_preprocessing(
    energies: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = torch.tensor(
        [radial_ordering(positions[i]) for i in range(positions.shape[0])],
        device=energies.device,
        dtype=torch.bool,
    )

    return energies[valid_mask], positions[valid_mask]
