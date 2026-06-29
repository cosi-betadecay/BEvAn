import itertools

import torch

from utils.megalib_types import MSimEvent


def candidate_perms(r: int) -> torch.Tensor:
    """Candidate orderings for a size-``r`` subset, as positions into its sorted hits.

    Reproduces the reference construction ``for first in c: for second in c:
    (first, second, *sorted(rest))`` (size 3 = all 6 permutations; size 4 = first
    two free, last two sorted = 12), in that exact order — so the batched code emits
    rows in the same order the scalar loop did, leaving arm's argmin tie-breaking
    unchanged.

    Args:
        r (int): Subset size.

    Returns:
        ``(K, r)`` position-index orderings (K = 6 for r=3, 12 for r=4).
    """
    perms, seen = [], set()
    for first in range(r):
        for second in range(r):
            if first == second:
                continue
            o = (first, second, *sorted(set(range(r)) - {first, second}))
            if o not in seen:
                seen.add(o)
                perms.append(o)
    return torch.tensor(perms, dtype=torch.long)


def ckd_residual_batched(energies: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Max squared (cosφ_geo − cosφ_kin) over each sequence's internal scatters.

    Vectorized form of the per-sequence CKD residual: for every row (a candidate
    hit ordering) it scores how consistently the geometric and kinematic scatter
    angles agree along the sequence, reduced by the *worst* (max) vertex so a
    single inconsistent vertex can't be averaged away. Computed in float64 because
    the value only gates ordering selection (it is never a model feature), so the
    extra precision keeps the filter decisions stable without affecting the float32
    feature path.

    Args:
        energies: ``(M, r)`` per-hit energies in keV for M candidate orderings.
        positions: ``(M, r, 3)`` matching hit positions.

    Returns:
        ``(M,)`` worst-vertex squared residual per ordering; NaN where no internal
        scatter is usable (the reference's ``None``).
    """
    E = energies.to(torch.float64)
    P = positions.to(torch.float64)

    # W[i] = (summed energy still to be deposited after hit i) / 511, in 511-keV
    # units. W[:, -1] is 0 and is never read (no scatter is internal at the end).
    W = (E.sum(dim=1, keepdim=True) - E.cumsum(dim=1)) / 511.0  # (M, r)

    legs = P[:, 1:, :] - P[:, :-1, :]  # (M, r-1, 3) hit i -> i+1
    norm = legs.norm(dim=2)  # (M, r-1)
    valid_leg = norm >= 1e-9  # unit() returns None below this
    legs_unit = legs / norm.clamp_min(1e-12).unsqueeze(-1)

    # Internal scatter i (1..r-2) uses legs (i-1, i) and residual energies W[i-1], W[i].
    cos_geo = (legs_unit[:, :-1, :] * legs_unit[:, 1:, :]).sum(dim=2)  # (M, r-2)
    w_in = W[:, :-2]  # (M, r-2)
    w_out = W[:, 1:-1]  # (M, r-2)
    cos_kin = (1.0 + 1.0 / w_in - 1.0 / w_out).clamp(-1.0, 1.0)
    resid = (cos_geo - cos_kin) ** 2  # (M, r-2)

    valid = valid_leg[:, :-1] & valid_leg[:, 1:] & (w_in >= 1e-9) & (w_out >= 1e-9)
    # Worst-vertex aggregation: a correct Compton sequence should be consistent at
    # *every* internal vertex, so we take the max squared residual rather than the
    # mean (which can mask one bad vertex with a good one). Invalid vertices are
    # excluded via a -inf sentinel; an ordering with no usable vertex yields NaN.
    worst = torch.where(valid, resid, resid.new_full((), float("-inf"))).amax(dim=1)
    return torch.where(valid.any(dim=1), worst, worst.new_full((), float("nan")))


def energy_desc_order(subsets: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
    """Reorder each subset's hits by descending energy (stable on ties).

    Stable matches the reference ``sorted(..., reverse=True)``, which keeps the
    original ascending-index order for equal energies.

    Args:
        subsets: ``(S, r)`` hit indices, each row ascending.
        energies: ``(n_hits,)`` per-hit energies in keV.

    Returns:
        ``(S, r)`` hit indices reordered by descending energy.
    """
    order = torch.argsort(energies[subsets], dim=1, descending=True, stable=True)
    return torch.gather(subsets, 1, order)


def ckd_orderings(subsets: torch.Tensor, energies: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """CKD-consistent candidate orderings for a batch of same-size subsets.

    Expands each subset into its candidate orderings (via :func:`candidate_perms`), scores
    them with :func:`ckd_residual_batched`, and keeps those within
    :data:`CKD_ORDER_MAX` (NaN/no-scatter counts as consistent). Subsets with no
    consistent ordering fall back to their single most-consistent one. Rows are
    emitted subset-by-subset in template order, matching the reference.

    Args:
        subsets: ``(S, r)`` hit indices (r in {3, 4}), each row ascending.
        energies: ``(n_hits,)`` per-hit energies in keV.
        positions: ``(n_hits, 3)`` per-hit positions.

    Returns:
        ``(n_kept, r)`` surviving candidate orderings (hit indices).
    """
    r = subsets.shape[1]
    perm = candidate_perms(r)  # (K, r)
    cand = subsets[:, perm]  # (S, K, r)
    s, k, _ = cand.shape

    flat = cand.reshape(s * k, r)
    resid = ckd_residual_batched(energies[flat], positions[flat]).reshape(s, k)

    CKD_ORDER_MAX = 0.05  # keep an ordering if worst-vertex sq (cosφ_geo - cosφ_kin) <= this

    keep = torch.isnan(resid) | (resid <= CKD_ORDER_MAX)  # (S, K)
    no_consistent = ~keep.any(dim=1)  # (S,)
    fallback = resid.argmin(dim=1)  # finite for no_consistent rows (no NaN there)
    keep[no_consistent, fallback[no_consistent]] = True
    return cand[keep]  # (n_kept, r), row-major over (S, K)


def orderings_for_size(
    subsets: torch.Tensor,
    energies: torch.Tensor,
    positions: torch.Tensor,
    ordering: str,
) -> torch.Tensor:
    """Candidate orderings for one size group of subsets.

    Dispatches by subset size: singletons pass through; size 2 takes the
    E_1 >= E_2 ordering; sizes 3-4 keep their CKD-consistent orderings; larger
    sizes keep the subset plus its energy-sorted ordering. ``ordering == "energy"``
    bypasses CKD entirely and returns one energy-sorted ordering per subset
    (the no-CKD-order ablation).

    Args:
        subsets: ``(S, r)`` hit indices, each row ascending.
        energies: ``(n_hits,)`` per-hit energies in keV.
        positions: ``(n_hits, 3)`` per-hit positions.
        ordering: ``"ckd"`` or ``"energy"``.

    Returns:
        ``(n_kept, r)`` candidate orderings (hit indices).
    """
    r = subsets.shape[1]
    if r == 1:
        return subsets
    if ordering == "energy" or r == 2:
        # Size-2 has no CKD redundancy, so both modes use the E_1 >= E_2 ordering
        # (Boggs & Jean §3); in energy mode every size collapses to this.
        return energy_desc_order(subsets, energies)
    if r <= 4:
        return ckd_orderings(subsets, energies, positions)

    # r >= 5: emit the subset, plus its energy-sorted ordering when it differs.
    desc = energy_desc_order(subsets, energies)
    differs = ~(desc == subsets).all(dim=1)  # (S,)
    stacked = torch.stack([subsets, desc], dim=1)  # (S, 2, r)
    keep = torch.stack([torch.ones_like(differs), differs], dim=1)  # (S, 2)
    return stacked[keep]


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
    # the subset enumeration below. The device boundary is pushed down to the big
    # gathered combo tensors, which is the only GPU-shaped work here.
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

    # Build all hit combinations, vectorized per size group. itertools.combinations
    # generates each size's subsets (C-level, far faster than torch.combinations);
    # orderings_for_size then expands and CKD-filters the whole group as one batched
    # tensor op instead of a per-subset Python loop.
    max_r = min(n_hits, 7)  # Boggs & Jean (2000)
    idx_parts = []
    size_parts = []
    for r in range(1, max_r + 1):
        subsets = torch.tensor(list(itertools.combinations(range(n_hits), r)), dtype=torch.long)
        ordered = orderings_for_size(subsets, energies, positions, ordering)  # (n_kept, r)
        if r < max_r:
            pad = torch.full((ordered.shape[0], max_r - r), n_hits, dtype=torch.long)
            ordered = torch.cat([ordered, pad], dim=1)
        idx_parts.append(ordered)
        size_parts.append(torch.full((ordered.shape[0],), r, dtype=torch.long))

    idx = torch.cat(idx_parts, dim=0).to(device)
    sizes = torch.cat(size_parts, dim=0).to(device)

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
