import itertools
from typing import Any

import torch


def event_data_processing(event: Any) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return None, None

    # Load event data onto GPU
    energies = torch.tensor(
        [event.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

    positions = torch.tensor(
        [
            (
                event.GetHTAt(i).GetPosition().X(),
                event.GetHTAt(i).GetPosition().Y(),
                event.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    for r in range(1, n_hits + 1):
        for c in itertools.combinations(range(n_hits), r):
            combos.append(c)
            sizes.append(r)

    max_r = max(sizes)
    n_combo = len(combos)

    # Pad combinations with sentinel index n_hits (maps to appended NaN row)
    idx = torch.full((n_combo, max_r), n_hits, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # Append one all-NaN entry so padded indices gather an unmistakably invalid
    # sentinel instead of a potentially physical zero hit/position.
    energies_ext = torch.cat([energies, energies.new_full((1,), float("nan"))], dim=0)  # (n_hits + 1,)
    positions_ext = torch.cat([positions, positions.new_full((1, 3), float("nan"))], dim=0)  # (n_hits + 1, 3)

    energy_combo = energies_ext[idx]  # (n_combo, max_r)
    pos_combo = positions_ext[idx]  # (n_combo, max_r, 3)

    if energy_combo.numel() == 0 or pos_combo.numel() == 0:
        return None, None

    return energy_combo, pos_combo


def strip_padding_from_event(
    energies: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if energies.ndim != 1:
        raise ValueError(
            f"Expected 1D energies for a single event candidate, got shape {tuple(energies.shape)}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions shape (N, 3), got {tuple(positions.shape)}")
    if positions.shape[0] != energies.shape[0]:
        raise ValueError(
            "Energies and positions must have the same leading dimension, "
            f"got {energies.shape[0]} and {positions.shape[0]}"
        )

    valid = torch.isfinite(energies) & torch.all(torch.isfinite(positions), dim=1)
    return energies[valid], positions[valid]
