import itertools
import math
from typing import Any

import torch
from tqdm import tqdm


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

    if energies.numel() == 0 or positions.numel() == 0:
        return None, None

    return energies, positions


def iter_event_permutation_batches(
    energies: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int = 2048,
):
    """
    Yield full-length hit permutations in bounded-size batches.

    Each batch represents alternative interaction-order hypotheses for the same
    event. The full N! search space is preserved, but only `batch_size`
    permutations are materialized at a time.
    """
    n_hits = energies.shape[0]
    if n_hits == 0:
        return

    batch: list[tuple[int, ...]] = []
    total_permutations = math.factorial(n_hits)
    permutation_iter = tqdm(
        itertools.permutations(range(n_hits), n_hits),
        total=total_permutations,
        desc=f"Hit permutations (N={n_hits})",
        unit="perm",
        leave=False,
    )
    for perm in permutation_iter:
        batch.append(perm)
        if len(batch) == batch_size:
            idx = torch.tensor(batch, dtype=torch.long, device=energies.device)
            yield energies[idx], positions[idx]
            batch.clear()

    if batch:
        idx = torch.tensor(batch, dtype=torch.long, device=energies.device)
        yield energies[idx], positions[idx]
