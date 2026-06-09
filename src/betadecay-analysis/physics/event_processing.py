import itertools
from typing import Any

import torch


def event_data_processing(
    event_sim: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hits = event_sim.GetNHTs()

    if n_hits == 0:
        return None, None, None

    # Load event data onto GPU
    energies = torch.tensor(
        [event_sim.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

    positions = torch.tensor(
        [
            (
                event_sim.GetHTAt(i).GetPosition().X(),
                event_sim.GetHTAt(i).GetPosition().Y(),
                event_sim.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    # ARM-targeted subset augmentation (SSD / Boggs & Jean §3).
    #
    # For events with <=7 hits the enumeration below already emits EVERY subset,
    # and two of the three features are order-invariant and therefore already
    # globally optimal over that pool: delta_E is a sum, annihilation_angle is a
    # min over all pairwise inter-hit vectors. They cannot be improved by adding
    # candidates. `arm` is the exception: theta_geo reads the first/second hit as
    # the scatter origin and direction, and theta_kin splits energy as
    # E_2 = (subset total) - E_first, so arm depends on the *ordering* of a
    # subset. Yet the enumeration only ever emits the physically-arbitrary
    # ascending-index ordering, so arm's argmin never sees the kinematically
    # sensible orderings.
    #
    # Augment (do not remove): for each subset of size >=2, also emit the
    # SSD-preferred ordering with hits sorted by descending deposited energy
    # (E_1 > E_2 > ...), per Boggs & Jean §3's "prefer the order where the first
    # scatter deposits more energy". `arm` takes a min over candidates, so this
    # lets its argmin reach a physics-ordered version. delta_E/angle are
    # unchanged (order-invariant), so the reward is cleanly attributable to arm.
    # Bounded at <=2x combos (one extra ordering per subset, skipped when it
    # coincides with the ascending order) to avoid the combinatorial blow-up a
    # full (first, second) permutation would cause on high-multiplicity events.
    energies_cpu = energies.tolist()

    for r in range(1, min(n_hits + 1, 8)):  # Boggs & Jean (2000)
        for c in itertools.combinations(range(n_hits), r):
            combos.append(c)
            sizes.append(r)
            if r >= 2:
                ordered = tuple(sorted(c, key=lambda h: energies_cpu[h], reverse=True))
                if ordered != c:
                    combos.append(ordered)
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
