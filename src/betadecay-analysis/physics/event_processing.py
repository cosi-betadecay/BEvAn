import itertools
from typing import Any

import ROOT as M
import torch


def event_data_processing(
    event_tra: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hits = event_tra.GetNHits()

    if event_tra.GetType() == M.MPhysicalEvent.c_Photo:
        E = float(event_tra.GetEnergy())
        P = event_tra.GetPosition()
        energies = torch.tensor([[E]], dtype=torch.float32, device=device)  # (1, 1)
        positions = torch.tensor(
            [[(P.X(), P.Y(), P.Z())]],
            dtype=torch.float32,
            device=device,  # (1, 1, 3)
        )
        sizes = torch.tensor([1], device=device)  # (1,)
        return energies, positions, sizes

    if n_hits == 0:
        return None, None, None

    # Load event data onto GPU
    energies = torch.tensor(
        [event_tra.GetHit(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

    positions = torch.tensor(
        [
            (
                event_tra.GetHit(i).GetPosition().X(),
                event_tra.GetHit(i).GetPosition().Y(),
                event_tra.GetHit(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    for r in range(1, min(n_hits + 1, 8)):  # Boggs & Jean (2000)
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
        return None, None, None

    return energy_combo, pos_combo, sizes
