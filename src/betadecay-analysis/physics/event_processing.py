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
    positions_cpu = positions.tolist()

    # CKD-consistency ORDERING filter (Boggs & Jean §2). Removing whole subsets
    # is near-zero-sum (V1 + V2 iters 5-6): it also raises the order-invariant
    # delta_E/angle mins, so every FP removed costs a TP. Instead, keep every
    # subset and filter only the ORDERINGS fed to `arm` (the sole order-sensitive
    # feature). For size 3-4 subsets we emit only kinematically-consistent
    # orderings — those whose geometric scatter angles match the Compton-formula
    # angles. A genuine Compton track has such an ordering (kept -> arm low,
    # signal preserved); a background coincidence's spuriously-low-arm ordering
    # is usually inconsistent (dropped -> arm rises -> fewer FP). delta_E/angle
    # are untouched because every subset still emits at least one ordering.
    CKD_ORDER_MAX = 0.05  # keep an ordering if mean sq (cosφ_geo − cosφ_kin) <= this

    def _unit(a, b):
        v = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        if n < 1e-9:
            return None
        return (v[0] / n, v[1] / n, v[2] / n)

    def _ckd_residual(seq):
        """Mean squared (cosφ_geo − cosφ_kin) over `seq`'s internal scatters."""
        N = len(seq)
        Es = [energies_cpu[h] for h in seq]
        P = [positions_cpu[h] for h in seq]
        W = [0.0] * N  # residual photon energy after hit i, units of 511 keV
        suffix = 0.0
        for i in range(N - 1, -1, -1):
            W[i] = suffix / 511.0
            suffix += Es[i]
        dirs = [_unit(P[i], P[i + 1]) for i in range(N - 1)]
        resids = []
        for i in range(1, N - 1):
            if dirs[i - 1] is None or dirs[i] is None:
                continue
            w_in, w_out = W[i - 1], W[i]
            if w_in < 1e-9 or w_out < 1e-9:
                continue
            cos_geo = sum(dirs[i - 1][k] * dirs[i][k] for k in range(3))
            cos_kin = 1.0 + 1.0 / w_in - 1.0 / w_out
            cos_kin = max(-1.0, min(1.0, cos_kin))
            resids.append((cos_geo - cos_kin) ** 2)
        if not resids:
            return None
        return sum(resids) / len(resids)

    def _orderings(c):
        """Candidate orderings of subset `c` that expose distinct `arm` values.

        `arm` depends only on which hit is FIRST (it sets E_2 = total - E_first
        and the geometric scatter origin) and which is SECOND (the geometric
        direction). For short subsets we enumerate the (first, second) orderings;
        for size 3-4 we then keep only the CKD-consistent ones so `arm`'s argmin
        cannot exploit a spurious low-arm ordering of a background subset. Size 2
        has no internal scatter (no CKD redundancy) so both orderings are kept.
        Size >= 5 keeps iter-1's bounded ascending+SSD pair (orderings there
        never change arm's per-event min anyway — V2 iter3). delta_E/angle are
        order-invariant and every subset still emits >=1 ordering, so they are
        untouched and the reward is cleanly attributable to arm.
        """
        r = len(c)
        if r < 2:
            return [c]
        if r <= 4:
            cset = set(c)
            cand = []
            seen = set()
            for first in c:
                for second in c:
                    if first == second:
                        continue
                    o = (first, second) + tuple(sorted(cset - {first, second}))
                    if o not in seen:
                        seen.add(o)
                        cand.append(o)
            if r == 2:
                return cand  # no CKD redundancy for 2-hit subsets
            scored = [(o, _ckd_residual(o)) for o in cand]
            consistent = [o for o, res in scored if res is None or res <= CKD_ORDER_MAX]
            if consistent:
                return consistent
            # No consistent ordering: emit only the most-consistent one, so the
            # subset still feeds delta_E/angle without giving arm a spurious pick.
            return [min(scored, key=lambda x: (x[1] if x[1] is not None else 0.0))[0]]
        ordered = tuple(sorted(c, key=lambda h: energies_cpu[h], reverse=True))
        return [c] if ordered == c else [c, ordered]

    for r in range(1, min(n_hits + 1, 8)):  # Boggs & Jean (2000)
        for c in itertools.combinations(range(n_hits), r):
            for o in _orderings(c):
                combos.append(o)
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
