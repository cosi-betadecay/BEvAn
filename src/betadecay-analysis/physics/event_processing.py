import torch

from utils.megalib_types import MSimEvent


def unit(a: list[float], b: list[float]) -> tuple[float, float, float] | None:
    """Unit vector from point ``a`` to point ``b``, or None if they coincide.

    Args:
        a (list[float]): First 3D point.
        b (list[float]): Second 3D point.

    Returns:
        tuple[float, float, float] | None: The unit vector from ``a`` to ``b``, or None if they coincide.
    """
    v = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
    if n < 1e-9:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def ckd_residual(
    seq: tuple[int, ...],
    energies_cpu: list[float],
    positions_cpu: list[list[float]],
) -> float | None:
    """Mean squared (cosφ_geo − cosφ_kin) over ``seq``'s internal scatters.

    Args:
        seq (tuple[int, ...]): Candidate hit ordering — the hypothesized scatter
            sequence, given as hit indices into ``energies_cpu``/``positions_cpu``.
        energies_cpu (list[float]): Hit energies in keV, indexed by hit.
        positions_cpu (list[list[float]]): Hit positions in 3D space, indexed by hit.

    Returns:
        float | None: The mean squared residual or None if not applicable.
    """
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


def orderings(
    energies_cpu: list[float],
    positions_cpu: list[list[float]],
    c: tuple[int, ...],
    ordering: str,
    ckd_order_max: float,
) -> list[tuple[int, ...]]:
    """Candidate hit orderings for subset ``c``, filtered by CKD consistency.

    Size-2 subsets return the energy-ordered (E_1 >= E_2) pair; size 3-4
    keep orderings whose mean CKD residual is within ``ckd_order_max`` (or
    the single most-consistent one if none qualify); larger subsets return
    the input plus its energy-sorted ordering. With ``ordering == "energy"``
    the CKD logic is bypassed and every subset returns its single energy-sorted
    ordering (the no-CKD-order ablation).

    Args:
        energies_cpu (list[float]): Hit energies in keV, indexed by hit.
        positions_cpu (list[list[float]]): Hit positions in 3D space, indexed by hit.
        c (tuple[int, ...]): Hit subset (indices) to order.
        ordering (str): ``"ckd"`` keeps CKD-consistent orderings; ``"energy"``
            bypasses the CKD logic and returns one energy-sorted ordering.
        ckd_order_max (float): Keep an ordering whose mean squared
            (cosφ_geo − cosφ_kin) residual is within this threshold.

    Returns:
        list[tuple[int, ...]]: Candidate orderings (hit-index tuples) for ``c``.
    """
    r = len(c)
    if r < 2:
        return [c]
    if ordering == "energy":
        return [tuple(sorted(c, key=lambda h: energies_cpu[h], reverse=True))]
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
        scored = [(o, ckd_residual(o, energies_cpu, positions_cpu)) for o in cand]
        consistent = [o for o, res in scored if res is None or res <= ckd_order_max]
        if consistent:
            return consistent
        # No consistent ordering: emit only the most-consistent one, so the
        # subset still feeds delta_E/angle without giving arm a spurious pick.
        return [min(scored, key=lambda x: x[1] if x[1] is not None else 0.0)[0]]
    ordered = tuple(sorted(c, key=lambda h: energies_cpu[h], reverse=True))
    return [c] if ordered == c else [c, ordered]


def event_data_processing(
    event_sim: MSimEvent,
    ordering: str = "ckd",
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Expand one simulated event into per-hit-subset energy and position tensors.

    Enumerates hit subsets up to size 7 (Boggs & Jean 2000), orders each by
    Compton-kinematic-discrepancy consistency, and pads them to a fixed width
    with a NaN sentinel row so downstream gathers stay rectangular.

    Args:
        event_sim: A MEGAlib simulated event with measured hits.
        ordering: Hit-subset ordering — ``"ckd"`` (default; keep CKD-consistent
            orderings) or ``"energy"`` (a single energy-sorted ordering per subset,
            the no-CKD-order ablation).

    Returns:
        ``(energies, positions, sizes)`` shaped ``(n_combo, max_r)``,
        ``(n_combo, max_r, 3)``, and ``(n_combo,)``; or ``(None, None, None)``
        for a hitless event.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hits = event_sim.GetNHTs()

    if n_hits == 0:
        return None, None, None

    # Load per-hit event data on CPU: it is tiny (n_hits values) and only feeds
    # the serial CPU subset enumeration below. The device boundary is pushed down
    # to the big gathered combo tensors, which is the only GPU-shaped work here.
    energies = torch.tensor(
        [event_sim.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device="cpu",
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
        device="cpu",
    )

    # Build all hit combinations. torch.combinations generates each size's
    # subsets in the same lexicographic order as itertools.combinations, so the
    # row order — and thus arm's argmin tie-breaking downstream — is unchanged.
    # The per-subset CKD ordering stays in orderings(); vectorizing that is a
    # separate step.
    combos = []
    sizes = []

    energies_cpu = energies.tolist()
    positions_cpu = positions.tolist()

    CKD_ORDER_MAX = 0.05  # keep an ordering if mean sq (cosφ_geo - cosφ_kin) <= this

    hit_idx = torch.arange(n_hits)
    for r in range(1, min(n_hits + 1, 8)):  # Boggs & Jean (2000)
        subsets = hit_idx.unsqueeze(1) if r == 1 else torch.combinations(hit_idx, r=r)
        for c in subsets.tolist():
            for o in orderings(energies_cpu, positions_cpu, tuple(c), ordering, CKD_ORDER_MAX):
                combos.append(o)
                sizes.append(r)

    max_r = max(sizes)

    # Pad each ordering to max_r with sentinel index n_hits (maps to the appended
    # NaN row) in a single tensor build instead of a per-combo scatter.
    padded = [list(o) + [n_hits] * (max_r - len(o)) for o in combos]
    idx = torch.tensor(padded, dtype=torch.long, device=device)
    sizes = torch.tensor(sizes, device=device)

    # Append one all-NaN entry so padded indices gather an unmistakably invalid
    # sentinel instead of a potentially physical zero hit/position. Move to the
    # device here (alongside idx) so the per-combo physics reductions run batched
    # over all n_combo rows — the actual GPU-parallel work.
    energies_ext = torch.cat([energies, energies.new_full((1,), float("nan"))], dim=0).to(
        device
    )  # (n_hits + 1,)
    positions_ext = torch.cat([positions, positions.new_full((1, 3), float("nan"))], dim=0).to(
        device
    )  # (n_hits + 1, 3)

    energy_combo = energies_ext[idx]  # (n_combo, max_r)
    pos_combo = positions_ext[idx]  # (n_combo, max_r, 3)

    return energy_combo, pos_combo, sizes
