import itertools
from typing import Any

import omegaconf
import torch

from mathematics.calculations import calculate_tolerance
from physics.posterior import posterior_bdecay
from physics.preprocessing.preprocesser import preprocesser


def detected_511_event(
    ref_energy: float,
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return 0.0, 0.0

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

    # Pad combinations with -1
    idx = torch.full((n_combo, max_r), -1, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # idx: (n_combo, max_r), padded with -1

    energy_combo = energies[idx]  # (n_combo, max_r)
    pos_combo = positions[idx]  # (n_combo, max_r, 3)

    mask = idx >= 0
    energy_combo = torch.where(mask, energy_combo, torch.zeros_like(energy_combo))
    pos_combo = torch.where(mask[..., None], pos_combo, torch.zeros_like(pos_combo))

    new_energy_combo, new_pos_combo = preprocesser(energy_combo, pos_combo, tolerance, cfg.preprocessing)

    if new_energy_combo.numel() == 0 or new_pos_combo.numel() == 0:
        return False, -1

    best_score_bdecay = posterior_bdecay(
        new_energy_combo,
        new_pos_combo,
        ref_energy,
        tolerance,
        cfg.likelihoods,
        sizes,
    )

    best_score_bg = 1 - best_score_bdecay

    return best_score_bdecay, best_score_bg
