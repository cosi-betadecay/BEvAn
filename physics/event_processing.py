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

    # Build all full-length hit orderings. Each row represents the same event
    # under a different interaction-order hypothesis, without introducing
    # shorter subsets or zero-padded pseudo-hits.
    permutations = list(itertools.permutations(range(n_hits), n_hits))
    idx = torch.tensor(permutations, dtype=torch.long, device=device)

    energy_combo = energies[idx]  # (n_perm, n_hits)
    pos_combo = positions[idx]  # (n_perm, n_hits, 3)

    if energy_combo.numel() == 0 or pos_combo.numel() == 0:
        return None, None

    return energy_combo, pos_combo
