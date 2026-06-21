import numpy as _np
import omegaconf
import torch
from tqdm import tqdm

############################################
# Annihilation Angle
############################################

# PRODUCTION statistic (when per-hit energies are supplied): the vertex-anchored
# back-to-back annihilation score, ``_back_to_back_score``. A real e+ annihilation
# emits two 511 keV photons in opposite directions from a common vertex; the sim
# truth (data/Activation.sim) shows that vertex sits within ~2 mm of a hit 95% of
# the time, so each hit is tried as the vertex and the two most anti-parallel arm
# directions from it give a back-to-back cosine (≈ -1 for a real pair). A single
# Compton photon can deposit at most ~511 keV total, so a candidate whose two-arm
# energy does not exceed that ceiling is a single-photon collinear fake and is
# rejected; survivors pay a deficit-only penalty rewarding a two-arm total toward
# 1022 keV (partial deposits are common — per-arm energy is NOT gated at 511).
# Lower score = more annihilation-like. NaN where no subset has the structure
# (only ~29% of annihilations deposit both photons — this is a high-purity
# minority signal; delta_E carries the single-photon 511 keV events).
#
# FALLBACK (no energies): the pure-geometry global minimum pairwise inter-hit
# cosine, optionally converted by ``normalize_min_cos`` to its left-tail
# percentile under a matched-n null (scripts/build_angle_null.py,
# data/angle_null_table.npz) so the raw min does not merely track hit count.


def load_null_table(path: str) -> dict:
    """Load the per-multiplicity null built by scripts/build_angle_null.py."""
    data = _np.load(path)
    ns = data["ns"]
    return {
        "n_min": int(ns[0]),
        "n_max": int(ns[-1]),
        "q_levels": data["q_levels"],
        "q_values": data["q_values"],
    }


def normalize_min_cos(raw: torch.Tensor, n_hits: int, null_table: dict) -> torch.Tensor:
    """Left-tail percentile of an observed min-cos under the matched-n null.

    ``null_table`` carries the per-multiplicity quantile grid produced by
    ``scripts/build_angle_null.py``: keys ``n_min``/``n_max`` (ints), ``q_levels``
    (Q,) ascending in [0, 1], and ``q_values`` (K, Q) the null min-cos at each
    (n, level), ascending along Q. ``n_hits`` is clamped to [n_min, n_max]
    (high-n nulls are already pinned at -1, so clamping loses nothing). Returns
    a 1-element tensor in [0, 1]; NaN in -> NaN out.
    """
    raw_f = float(raw.reshape(-1)[0])
    if raw_f != raw_f:  # NaN
        return raw.new_tensor(float("nan")).reshape(1)
    n_min, n_max = null_table["n_min"], null_table["n_max"]
    idx = min(max(n_hits, n_min), n_max) - n_min
    q_values = null_table["q_values"][idx]
    q_levels = null_table["q_levels"]
    pct = float(_np.interp(raw_f, q_values, q_levels))
    return raw.new_tensor(pct).reshape(1)


def _per_subset_min_cosines(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-subset most-back-to-back cosine.

    Mirrors ``all_vector_cosines_matrix_calculation`` but returns one value per
    subset (shape ``(B,)``) instead of collapsing the whole batch with a global
    ``amin``. Taking ``amin`` over the returned vector reproduces the global
    minimum exactly, so this is a behavior-preserving refactor at zero penalty;
    keeping the values per-subset is what lets the energy penalty be applied to
    each candidate before the minimum is taken.
    """
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    b, n = positions.shape[0], positions.shape[1]
    if n < 3:
        return positions.new_full((b,), float("nan"))

    deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    vectors = deltas[:, i, j, :]  # (B, M, 3), M = N choose 2
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

    cosine_matrix = vectors @ vectors.transpose(-1, -2)  # (B, M, M)
    m = vectors.shape[1]
    a, c = torch.triu_indices(m, m, offset=1, device=positions.device)
    cosines = cosine_matrix[:, a, c]  # (B, K)

    return cosines.amin(dim=1)  # (B,)


def all_vector_cosines_matrix_calculation(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
        unbatched = True
    elif positions.ndim == 3:
        unbatched = False
    else:
        raise ValueError(f"Expected positions shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}")

    n = positions.shape[1]
    if n < 3:
        empty = positions.new_empty((positions.shape[0], 0))
        return empty.squeeze(0) if unbatched else empty

    deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    vectors = deltas[:, i, j, :]  # (B, M, 3), M = N choose 2
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

    cosine_matrix = vectors @ vectors.transpose(-1, -2)  # (B, M, M)
    m = vectors.shape[1]
    a, b = torch.triu_indices(m, m, offset=1, device=positions.device)
    cosines = cosine_matrix[:, a, b]  # (B, K)

    return cosines.amin().reshape(-1)


def all_vector_cosines_blockwise(
    positions: torch.Tensor,
    eps: float = 1e-12,
    combo_block_size: int = 1024,
    vec_block_size: int = 128,
) -> torch.Tensor:
    if positions.ndim != 3:
        raise ValueError(f"Expected positions shape (B, N, 3), got {tuple(positions.shape)}")

    n = positions.shape[1]
    if n < 3:
        return positions.new_empty((positions.shape[0], 0))

    total_b = positions.shape[0]
    global_best = torch.full((total_b,), 1.0, dtype=positions.dtype, device=positions.device)

    pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=positions.device)

    combo_starts = range(0, total_b, combo_block_size)
    for b0 in tqdm(combo_starts, desc="Annihilation angle blocks", unit="block", leave=False):
        b1 = min(b0 + combo_block_size, total_b)
        pos_chunk = positions[b0:b1]  # (Bc, N, 3)

        deltas = pos_chunk[:, :, None, :] - pos_chunk[:, None, :, :]  # (Bc, N, N, 3)
        vectors = deltas[:, pair_i, pair_j, :]  # (Bc, M, 3)
        vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

        bsz, m, _ = vectors.shape
        chunk_best = torch.full((bsz,), 1.0, dtype=vectors.dtype, device=vectors.device)

        for i0 in range(0, m, vec_block_size):
            i1 = min(i0 + vec_block_size, m)
            vi = vectors[:, i0:i1, :]  # (Bc, bi, 3)

            for j0 in range(i0, m, vec_block_size):
                j1 = min(j0 + vec_block_size, m)
                vj = vectors[:, j0:j1, :]  # (Bc, bj, 3)

                block = torch.matmul(vi, vj.transpose(-1, -2))  # (Bc, bi, bj)

                if i0 == j0:
                    bi = i1 - i0
                    bj = j1 - j0
                    diag = torch.eye(bi, bj, dtype=torch.bool, device=block.device).unsqueeze(0)
                    block = block.masked_fill(diag, 1.0)

                block_min = block.amin(dim=(1, 2))
                chunk_best = torch.minimum(chunk_best, block_min)

        global_best[b0:b1] = chunk_best

    return global_best.amin().reshape(-1)


# --- Back-to-back annihilation score (vertex-anchored, energy-gated) ----------
# Thresholds are derived from data/Activation.sim truth (see project memory):
B2B_E_FLOOR = 511.0 - 3.0 * (2.25 / 2.355)  # ~508 keV: a single Compton photon
#                                             cannot deposit more than this total
B2B_E_TARGET = 1022.0  # two fully-absorbed 511 keV photons
B2B_SIGMA_E = 250.0  # deficit-tolerant energy scale (partial deposits dominate)
B2B_LAMBDA = 0.3  # energy-penalty strength, in cosine units


def _per_subset_back_to_back(
    positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Vertex-anchored back-to-back annihilation score, one value per subset.

    For each hit taken as the annihilation vertex, the two most anti-parallel
    arm directions from that vertex give a back-to-back cosine (≈ -1 for a real
    pair). The two-arm energy (all non-vertex hits — the split itself does not
    change their sum) must exceed ``B2B_E_FLOOR`` or the candidate is a
    single-photon collinear fake and is rejected; survivors add a deficit-only
    Gaussian penalty toward ``B2B_E_TARGET``. The subset's score is the best
    (lowest) vertex. Lower = more annihilation-like.

    Args:
        positions: ``(B, n, 3)`` hit positions for B subsets of size n.
        energies: ``(B, n)`` per-hit deposited energy.

    Returns:
        ``(B,)`` scores; ``+inf`` for a subset with no gate-passing vertex
        (also where ``n < 3``, or for NaN-padded subsets).
    """
    B, n, _ = positions.shape
    if n < 3:
        return positions.new_full((B,), float("inf"))
    inf = positions.new_tensor(float("inf"))
    total_E = energies.sum(dim=1)  # (B,)
    best = positions.new_full((B,), float("inf"))
    for v in range(n):
        others = [k for k in range(n) if k != v]
        D = positions[:, others, :] - positions[:, v : v + 1, :]  # (B, m, 3)
        D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min(eps)
        C = D @ D.transpose(-1, -2)  # (B, m, m)
        m = len(others)
        eye = torch.eye(m, dtype=torch.bool, device=positions.device).unsqueeze(0)
        cos_bb = C.masked_fill(eye, float("inf")).amin(dim=(1, 2))  # most anti-parallel
        e_arms = total_E - energies[:, v]  # two-arm energy (vertex excluded)
        deficit = (B2B_E_TARGET - e_arms).clamp_min(0.0)
        e_term = B2B_LAMBDA * (1.0 - torch.exp(-(deficit**2) / (2.0 * B2B_SIGMA_E**2)))
        score = torch.where(e_arms > B2B_E_FLOOR, cos_bb + e_term, inf)
        best = torch.minimum(best, score)
    return best


def _back_to_back_score(
    positions: torch.Tensor,
    energies: torch.Tensor,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-event back-to-back score over the reconstructed subset pool.

    Mirrors the sized reduction of ``delta_E``/``arm``: group combos by subset
    size, score each, and return the per-event best (lowest = most
    annihilation-like), or NaN when no subset has >=3 hits with a gate-passing
    vertex.
    """
    if sizes is None:
        P = positions if positions.ndim == 3 else positions.unsqueeze(0)
        E = energies if energies.ndim == 2 else energies.unsqueeze(0)
        score = _per_subset_back_to_back(P, E)
        finite = score[torch.isfinite(score)]
        return (finite.amin() if finite.numel() else positions.new_tensor(float("nan"))).reshape(1)

    if positions.ndim != 3 or energies.ndim != 2 or sizes.ndim != 1:
        raise ValueError(
            "Sized back-to-back expects positions (B, N, 3), energies (B, N), sizes (B,), "
            f"got {tuple(positions.shape)}, {tuple(energies.shape)}, {tuple(sizes.shape)}"
        )
    if positions.shape[0] != sizes.shape[0] or energies.shape[0] != sizes.shape[0]:
        raise ValueError(
            "Batch dimension and sizes length must match, "
            f"got {tuple(positions.shape)}, {tuple(energies.shape)}, {tuple(sizes.shape)}"
        )

    outputs = []
    for size in torch.unique(sizes, sorted=True):
        size_int = int(size.item())
        if size_int < 3:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        group = sizes == size
        score = _per_subset_back_to_back(positions[group, :size_int, :], energies[group, :size_int])
        finite = score[torch.isfinite(score)]
        if finite.numel() == 0:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        outputs.append(finite.amin().reshape(1))

    stacked = torch.stack(outputs).flatten()
    finite = stacked[torch.isfinite(stacked)]
    if finite.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)
    return finite.amin().reshape(1)


def annihilation_angle(
    positions: torch.Tensor,
    n_hits: int | None = None,
    sizes: torch.Tensor | None = None,
    energies: torch.Tensor | None = None,
    null_table: dict | None = None,
) -> torch.Tensor:
    # PRODUCTION path: with per-hit energies, return the vertex-anchored
    # back-to-back annihilation score (geometry + single-photon energy gate).
    if energies is not None:
        return _back_to_back_score(positions, energies, sizes=sizes)

    # FALLBACK (no energies): pure-geometry min-cos, optionally null-normalized
    # to its left-tail percentile under the matched-``n_hits`` null.
    raw = _annihilation_min_cos(positions, n_hits=n_hits, sizes=sizes)
    if null_table is None:
        return raw
    if n_hits is None:
        n_hits = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    return normalize_min_cos(raw, n_hits, null_table)


def _annihilation_min_cos(
    positions: torch.Tensor,
    n_hits: int | None = None,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw global minimum pairwise inter-hit cosine (no energy, no normalization)."""
    if sizes is not None:
        if positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized annihilation_angle expects positions shape (B, N, 3) and sizes shape (B,), "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )
        if positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Positions batch dimension and sizes length must match, "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            group = sizes == size
            if size_int < 3:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            group_positions = positions[group, :size_int, :]
            outputs.append(_annihilation_min_cos(group_positions, n_hits=size_int))

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    if n_hits is None:
        n_hits = positions.shape[0] if positions.ndim == 2 else positions.shape[1]

    if n_hits <= 14:
        cosines = all_vector_cosines_matrix_calculation(positions)
    else:
        blockwise_positions = positions.unsqueeze(0) if positions.ndim == 2 else positions
        cosines = all_vector_cosines_blockwise(blockwise_positions)

    if cosines.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)

    angle_diffs = torch.abs(cosines + 1.0)

    return cosines[torch.argmin(angle_diffs)].reshape(1)


############################################
# Angular Resolution Measure (ARM)
############################################


# m_e c^2 (keV): the β⁺ hypothesis incoming-photon energy for the kinematic angle.
ARM_ELECTRON_MASS_KEV = 511.0


def _per_subset_arm(positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Vertex-anchored, source-independent Compton-consistency (ARM), per subset.

    The far-field source direction does not exist for volume-distributed
    activation, so the incoming direction is reconstructed PER EVENT: for every
    (vertex v, first-scatter i, second-scatter j) triple of hits, the geometric
    scatter angle at i is ``θ_geo = ∠(P_i − P_v, P_j − P_i)`` and the kinematic
    angle is the Compton angle for a 511 keV photon depositing ``E_i`` at the
    first scatter (E_out = ΣE − E_i). ARM is the smallest ``|θ_geo − θ_kin|`` over
    all triples — the most Compton-consistent emission→scatter→scatter reading the
    hits admit. Fully blind: only hit positions/energies, the 511 keV hypothesis,
    and a min over observable triples — no truth, no source direction.

    Args:
        positions: ``(B, n, 3)`` hit positions for B subsets of size n.
        energies: ``(B, n)`` per-hit deposited energy.

    Returns:
        ``(B,)`` ARM (radians); ``+inf`` where n < 3 or no valid triple exists.
    """
    B, n, _ = positions.shape
    if n < 3:
        return positions.new_full((B,), float("inf"))

    tot = energies.sum(dim=1)  # (B,)
    deltas = positions[:, None, :, :] - positions[:, :, None, :]  # (B, a, b, 3) = P_b − P_a
    D = deltas / torch.linalg.norm(deltas, dim=-1, keepdim=True).clamp_min(eps)
    cos_geo = torch.einsum("bvid,bijd->bvij", D, D).clamp(-1.0, 1.0)  # ∠(v→i, i→j)
    theta_geo = torch.arccos(cos_geo)  # (B, v, i, j)

    e_out = tot[:, None] - energies  # (B, i): outgoing energy after first scatter
    valid_i = e_out > eps
    cos_kin = 1.0 - ARM_ELECTRON_MASS_KEV * (1.0 / e_out.clamp_min(eps) - 1.0 / ARM_ELECTRON_MASS_KEV)
    theta_kin = torch.arccos(cos_kin.clamp(-1.0, 1.0))  # (B, i)

    arm_vals = (theta_geo - theta_kin[:, None, :, None]).abs()  # (B, v, i, j)

    idx = torch.arange(n, device=positions.device)
    distinct = (
        (idx[:, None, None] != idx[None, :, None])
        & (idx[None, :, None] != idx[None, None, :])
        & (idx[:, None, None] != idx[None, None, :])
    )  # (v, i, j) all distinct
    bad = ~distinct[None] | ~valid_i[:, None, :, None] | ~torch.isfinite(arm_vals)
    arm_vals = arm_vals.masked_fill(bad, float("inf"))
    return arm_vals.amin(dim=(1, 2, 3))  # (B,)


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    reconstructed_unit_vector,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-event vertex-anchored Compton-consistency score over the subset pool.

    Replaces the far-field ARM (which assumed a single source direction — invalid
    for volume-distributed activation, where it discriminates at chance). ``cfg``
    and ``reconstructed_unit_vector`` are accepted for call compatibility but
    UNUSED. Returns the per-event best (lowest) ARM, or NaN when no subset has a
    valid >=3-hit triple. Note: needs >=3 hits, so 2-hit events get NaN here and
    fall to the ΔE-only bucket.
    """
    if sizes is not None:
        if energies.ndim != 2 or positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized arm expects energies shape (B, N), positions shape (B, N, 3), and sizes shape (B,), "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0] or positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies/positions batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            if size_int < 3:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            group = sizes == size
            score = _per_subset_arm(positions[group, :size_int, :], energies[group, :size_int])
            finite = score[torch.isfinite(score)]
            if finite.numel() == 0:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            outputs.append(finite.amin().reshape(1))

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    P = positions if positions.ndim == 3 else positions.unsqueeze(0)
    E = energies if energies.ndim == 2 else energies.unsqueeze(0)
    score = _per_subset_arm(P, E)
    finite = score[torch.isfinite(score)]
    return finite.amin().reshape(1) if finite.numel() else positions.new_tensor(float("nan")).reshape(1)


############################################
# Energy
############################################


def delta_E(
    energies: torch.Tensor, ref_energy: float = 511.0, sizes: torch.Tensor | None = None
) -> torch.Tensor:
    if sizes is not None:
        if energies.ndim != 2 or sizes.ndim != 1:
            raise ValueError(
                "Sized delta_E expects energies shape (B, N) and sizes shape (B,), "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        seq = torch.arange(energies.shape[1], device=energies.device).unsqueeze(0)
        valid = seq < sizes.unsqueeze(1)
        E = torch.where(valid, energies, torch.zeros_like(energies)).sum(dim=1)
    else:
        E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    delta_E = torch.abs(E - ref_energy)

    return delta_E.min().reshape(1)
