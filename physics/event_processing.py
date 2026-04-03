import itertools
from typing import Any

import omegaconf
import torch

from mathematics.calculations import calculate_tolerance
from physics.preprocessing.preprocesser import preprocesser


def event_data_processing(
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tolerance = calculate_tolerance()
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

    # Pad combinations with sentinel index n_hits (maps to appended zero row)
    idx = torch.full((n_combo, max_r), n_hits, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # Append one all-zero entry so padded indices gather zeros directly.
    energies_ext = torch.cat([energies, energies.new_zeros(1)], dim=0)  # (n_hits + 1,)
    positions_ext = torch.cat([positions, positions.new_zeros((1, 3))], dim=0)  # (n_hits + 1, 3)

    energy_combo = energies_ext[idx]  # (n_combo, max_r)
    pos_combo = positions_ext[idx]  # (n_combo, max_r, 3)

    new_energy_combo, new_pos_combo = preprocesser(energy_combo, pos_combo, tolerance, cfg.preprocessing)

    if new_energy_combo.numel() == 0 or new_pos_combo.numel() == 0:
        return None, None

    return new_energy_combo, new_pos_combo
