import itertools

import torch
from utils.megalib_types import MSimEvent


def event_data_processing(
    event_sim: MSimEvent,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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

    energies_cpu = energies.tolist()
    positions_cpu = positions.tolist()

    CKD_ORDER_MAX = 0.05  # keep an ordering if mean sq (cosφ_geo - cosφ_kin) <= this

    def unit(a: list[float], b: list[float]) -> tuple[float, float, float] | None:
        v = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        if n < 1e-9:
            return None
        return (v[0] / n, v[1] / n, v[2] / n)

    def ckd_residual(seq: tuple[int, ...]) -> float | None:
        """Mean squared (cosφ_geo − cosφ_kin) over `seq`'s internal scatters."""
        N = len(seq)
        Es = [energies_cpu[h] for h in seq]
        P = [positions_cpu[h] for h in seq]
        W = [0.0] * N  # residual photon energy after hit i, units of 511 keV
        suffix = 0.0
        for i in range(N - 1, -1, -1):
            W[i] = suffix / 511.0
            suffix += Es[i]
        dirs = [unit(P[i], P[i + 1]) for i in range(N - 1)]
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

    def orderings(c: tuple[int, ...]) -> list[tuple[int, ...]]:

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
                    o = (first, second, *sorted(cset - {first, second}))
                    if o not in seen:
                        seen.add(o)
                        cand.append(o)
            if r == 2:
                # No CKD redundancy for 2-hit subsets, so use SSD instead
                # (Boggs & Jean §3): a real 2-site photon deposits more energy
                # in the first scatter, so feed arm only the E_1>=E_2 ordering.
                # This denies background 2-site subsets the reverse ordering that
                # can give a spurious low arm; delta_E/angle are unaffected.
                a, b = c
                return [(a, b) if energies_cpu[a] >= energies_cpu[b] else (b, a)]
            scored = [(o, ckd_residual(o)) for o in cand]
            consistent = [o for o, res in scored if res is None or res <= CKD_ORDER_MAX]
            if consistent:
                return consistent
            # No consistent ordering: emit only the most-consistent one, so the
            # subset still feeds delta_E/angle without giving arm a spurious pick.
            return [min(scored, key=lambda x: x[1] if x[1] is not None else 0.0)[0]]
        ordered = tuple(sorted(c, key=lambda h: energies_cpu[h], reverse=True))
        return [c] if ordered == c else [c, ordered]

    for r in range(1, min(n_hits + 1, 8)):  # Boggs & Jean (2000)
        for c in itertools.combinations(range(n_hits), r):
            for o in orderings(c):
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

    return energy_combo, pos_combo, sizes
