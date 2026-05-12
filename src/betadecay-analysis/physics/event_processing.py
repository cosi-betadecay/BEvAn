import itertools
from typing import Any

import torch


def lever_arm_mask(
    pos_combo: torch.Tensor,
    min_lever_arm_cm: float = 0.5,
) -> torch.Tensor:
    """Boolean keep-mask over subsets based on a minimum-lever-arm cut.

    Boggs & Jean (2000) §6.9 require a minimum distance between interaction
    sites for a Compton sequence to be usable: hits sitting too close
    together cannot be ordered reliably by any algorithm. Revan's
    ``MERCSRBayesian::CalculateMinLeverArm`` (Zoglauer, MEGAlib) defines
    this per consecutive triple in an already-chosen ordering as
    ``min(|Start - Central|, |Stop - Central|)``.

    At this stage we have not yet picked an ordering for each subset (that
    belongs to a later CKD step), so we use the conservative
    permutation-free analog: a subset is acceptable iff its minimum
    pairwise hit distance exceeds the threshold. Any subset failing this
    test cannot have a valid lever arm under *any* ordering.

    NaN-padded slots produce NaN distances and are excluded from the min;
    r=1 subsets (no pair) therefore evaluate to +inf and pass
    unconditionally.

    Positions are expected in MEGAlib's native units (cm). The 0.5 cm
    default sits comfortably above COSI's GeD spatial resolution
    (~0.5 mm in z, ~2 mm in xy) while staying well below the instrument
    scale.
    """
    _, max_r, _ = pos_combo.shape
    diff = pos_combo[:, :, None, :] - pos_combo[:, None, :, :]  # (n_combo, max_r, max_r, 3)
    dist = torch.linalg.norm(diff, dim=-1)  # (n_combo, max_r, max_r)
    eye = torch.eye(max_r, dtype=torch.bool, device=pos_combo.device).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    dist = torch.nan_to_num(dist, nan=float("inf"))
    min_dist = dist.amin(dim=(1, 2))  # (n_combo,)
    return min_dist >= min_lever_arm_cm


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

    # keep = lever_arm_mask(pos_combo)  # Boggs & Jean (2000) §6.9
    # energy_combo = energy_combo[keep]
    # pos_combo = pos_combo[keep]
    # sizes = sizes[keep]

    if energy_combo.numel() == 0 or pos_combo.numel() == 0:
        return None, None, None

    return energy_combo, pos_combo, sizes
