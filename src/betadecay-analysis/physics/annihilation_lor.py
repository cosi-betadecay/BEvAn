"""Blind line-of-response reconstruction of positron annihilation.

Reconstructs the e+e- annihilation geometry of an event from measured hits
only (positions + energies, no simulation truth), under the two-photon
hypothesis: the event contains two photon chains A and B, each depositing
~511 keV, emitted back-to-back from a common (unobserved) vertex.

The key identity that makes blind reconstruction over-constrained: both
photons leave the vertex along one line, so the first interaction points of
the two chains are collinear with the vertex. The line between them is the
line of response (LOR), and it supplies each chain's *incoming* direction —
information classic Compton-sequence ordering never has for the first
scatter. With the incoming direction and the known initial energy (511 keV),
the first scatter of every >=2-hit chain is checked twice:

  geometric:  cos(theta) = unit(first - other_first) . unit(second - first)
  kinematic:  cos(theta) = 2 - 511 / (511 - E_first)    [Compton, E_in=511]

A hypothesis is a (bipartition of hits into A/B, choice of first hit in
each), graded chi^2-style on per-chain energy residuals plus the
first-scatter cosine residuals. Hypothesis SELECTION uses an asymmetric
energy model (one chain anchored at 511 keV, deficits on the other graded
leniently — the truth census shows 90% of signal fully absorbs only one
photon, and lenient selection improves the axis median from 12.5 to 9.8
deg). The reported SCORE re-grades the selected hypothesis with the strict
symmetric 511/511 residual, which separates signal from background best
(TV 0.271 vs 0.188 on the Activation.sim harness).

What is fundamentally NOT recovered blindly: the vertex position along the
LOR (needs timing or a source prior) and events where one photon escaped
entirely (no second chain -> no LOR; the energy term then dominates and the
score is honestly large).

Conventions: energies in keV, positions in cm (matching MEGAlib hits).
Lower score = more annihilation-like. NaN when the event cannot express the
two-photon hypothesis at all (fewer than 2 hits) or is too large to
enumerate (> MAX_LOR_HITS, bounded by the 2^(N-1) bipartition count).
"""

from dataclasses import dataclass

import numpy as np

ELECTRON_MASS_KEV = 511.0
PHOTON_KEV = 511.0
# 2^(N-1) bipartitions; 12 hits -> 2047, well within budget. Events larger
# than this are vanishingly rare (mean multiplicity ~2.3) and get NaN.
MAX_LOR_HITS = 12
# Soft energy scale for the per-chain 511 keV residual. Much wider than the
# detector resolution (~1 keV sigma) on purpose: partial deposits should be
# *ranked*, not annihilated (V3 lesson — gentle penalties, no recall cliffs).
SIGMA_E_KEV = 30.0
# Deficit scale for the SECONDARY chain. The truth census shows 90% of
# signal events fully absorb only ONE photon; the second deposits partially
# (escape). Under-depositing is therefore physically normal and graded
# leniently, while any deposit ABOVE 511 keV stays tightly penalized — a
# single 511 photon cannot deposit more (V3 iter-7 one-sidedness).
SIGMA_E_DEFICIT_KEV = 250.0
# Soft scale for the (cos_geo - cos_kin) first-scatter residual. Absorbs
# Doppler acollinearity/broadening and finite position resolution.
SIGMA_COS = 0.05


@dataclass
class LORResult:
    """Best two-photon hypothesis found for one event.

    Selection and scoring are deliberately decoupled (both measured on the
    full Activation.sim truth harness): the hypothesis is CHOSEN with the
    asymmetric energy model (deficits lenient — 90% of real signal fully
    absorbs only one photon, so the true partition usually has a partial
    second chain; axis median error 12.5 -> 9.8 deg), but the reported
    ``score`` re-grades that hypothesis with the strict symmetric 511/511
    residual, which separates classes best (TV 0.271 vs 0.188 for the
    asymmetric score itself).
    """

    score: float  # symmetric-rescored residual; lower = more annihilation-like
    axis: np.ndarray | None  # unit vector of the reconstructed LOR, or None
    first_a: int | None  # hit index of chain A's first interaction
    first_b: int | None  # hit index of chain B's first interaction
    chain_a: np.ndarray | None  # bool mask of hits assigned to chain A
    energy_a: float  # deposited energy of chain A at the best hypothesis
    energy_b: float  # deposited energy of chain B at the best hypothesis
    selection_score: float = float("nan")  # asymmetric residual used to pick the hypothesis


def _first_scatter_cos_kin(first_energies: np.ndarray) -> np.ndarray:
    """Kinematic cosine of the first scatter for a 511 keV photon that
    deposits ``first_energies`` in its first interaction.

    cos(theta) = 1 - m_e c^2 (1/E_out - 1/E_in) with E_in = 511 keV and
    E_out = 511 - E_first. Deposits at/above 511 keV leave no outgoing
    photon (the chain should have been size 1) -> NaN, treated as a maximal
    residual by the caller. Values are clamped to [-1, 1] rather than
    rejected: V2 showed clamp-tolerance beats strict unphysical rejection
    under detector noise.
    """
    e_out = PHOTON_KEV - first_energies
    cos_kin = np.where(
        e_out > 1e-6,
        2.0 - PHOTON_KEV / np.maximum(e_out, 1e-6),
        np.nan,
    )
    return np.clip(cos_kin, -1.0, 1.0)


def _selection_energy_residual(e_a: float, e_b: float) -> float:
    """Asymmetric two-chain energy residual, used to SELECT the hypothesis.

    One chain (the PRIMARY — whichever fits better) must sit at ~511 keV,
    matching the event label's semantics (>=1 fully absorbed photon). The
    SECONDARY chain is graded one-sidedly: deficits (partial deposit /
    escaped photon — 90% of real signal) use the wide SIGMA_E_DEFICIT_KEV,
    while excesses above 511 keV stay on the tight scale because a single
    511 keV photon cannot deposit more. Both role assignments are tried and
    the better one used. Setting SIGMA_E_DEFICIT_KEV = SIGMA_E_KEV recovers
    the symmetric 511/511 residual.
    """

    def primary(e):
        return (e - PHOTON_KEV) ** 2 / (2.0 * SIGMA_E_KEV**2)

    def secondary(e):
        sigma = SIGMA_E_KEV if e > PHOTON_KEV else SIGMA_E_DEFICIT_KEV
        return (e - PHOTON_KEV) ** 2 / (2.0 * sigma**2)

    return min(primary(e_a) + secondary(e_b), primary(e_b) + secondary(e_a))


def _symmetric_energy_residual(e_a: float, e_b: float) -> float:
    """Strict 511/511 residual, used to SCORE the selected hypothesis."""
    return ((e_a - PHOTON_KEV) ** 2 + (e_b - PHOTON_KEV) ** 2) / (2.0 * SIGMA_E_KEV**2)


def _pairwise_units(positions: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """U[i, j] = unit vector from hit i to hit j (rows of zeros on the
    diagonal / for coincident hits)."""
    deltas = positions[None, :, :] - positions[:, None, :]  # (n, n, 3)
    norms = np.linalg.norm(deltas, axis=-1, keepdims=True)
    return deltas / np.maximum(norms, eps)


def _geo_residual_table(positions: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """G[k, i, j] = squared first-scatter inconsistency of the hypothesis
    'chain starting at hit i (incoming along k->i) scatters next to hit j'.

    Entries with i==j or k==i are invalid and set to +inf; NaN kinematics
    (first deposit >= 511 keV) become +inf as well, so they can never be
    chosen as a scatter ordering but a 1-hit chain (no geo term) remains
    available.
    """
    n = positions.shape[0]
    units = _pairwise_units(positions)  # U[a, b] = unit(b - a)
    cos_kin = _first_scatter_cos_kin(energies)  # (n,), depends on first hit i

    # cos_geo[k, i, j] = unit(i - k) . unit(j - i) = U[k, i] . U[i, j]
    cos_geo = np.einsum("kid,ijd->kij", units, units)
    resid = (cos_geo - cos_kin[None, :, None]) ** 2 / (2.0 * SIGMA_COS**2)

    resid = np.where(np.isnan(resid), np.inf, resid)
    idx = np.arange(n)
    resid[:, idx, idx] = np.inf  # j == i
    resid[idx, idx, :] = np.inf  # k == i
    return resid


def _chain_first_scatter_residuals(geo_table: np.ndarray, chain: np.ndarray, other_chain: np.ndarray) -> np.ndarray:
    """Best (min over second hit) first-scatter residual of ``chain`` for
    every (first-of-other-chain k, first-of-chain i) combination.

    Returns R[k, i] over the full hit index range with +inf where k is not
    in ``other_chain`` or i not in ``chain``. Chains of size 1 have no
    internal scatter to check -> residual 0 for their single valid i.
    """
    n = geo_table.shape[0]
    big = np.full((n, n), np.inf)
    chain_idx = np.flatnonzero(chain)
    other_idx = np.flatnonzero(other_chain)

    if chain_idx.size == 1:
        big[np.ix_(other_idx, chain_idx)] = 0.0
        return big

    # min over j in chain \ {i}: take the sub-table restricted to columns of
    # the chain; the j==i entries are already +inf in the table.
    sub = geo_table[np.ix_(other_idx, chain_idx, chain_idx)]  # (K, I, J)
    big[np.ix_(other_idx, chain_idx)] = sub.min(axis=2)
    return big


def annihilation_lor(positions, energies) -> LORResult:
    """Blind two-photon reconstruction of one event.

    Args:
        positions: (n, 3) hit positions [cm] (array-like; torch tensors ok).
        energies: (n,) deposited energies [keV].

    Returns:
        LORResult with the minimal hypothesis score (NaN if the event cannot
        be tested), the reconstructed annihilation axis, and the chain
        assignment behind it.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    n = positions.shape[0]

    if n < 2 or n > MAX_LOR_HITS:
        return LORResult(float("nan"), None, None, None, None, float("nan"), float("nan"))

    total_e = energies.sum()
    geo_table = _geo_residual_table(positions, energies)

    best_sel = np.inf
    best = None  # (geo_resid, i, k, chain_a, e_a, e_b)

    # Enumerate bipartitions with hit 0 pinned to chain A (A/B labels are
    # symmetric). mask bit b set -> hit b+1 belongs to A.
    for bits in range(2 ** (n - 1) - 1, -1, -1):
        chain_a = np.zeros(n, dtype=bool)
        chain_a[0] = True
        for b in range(n - 1):
            if bits >> b & 1:
                chain_a[b + 1] = True
        if chain_a.all():
            continue  # chain B empty
        chain_b = ~chain_a

        e_a = energies[chain_a].sum()
        e_b = total_e - e_a
        energy_resid = _selection_energy_residual(e_a, e_b)
        if energy_resid >= best_sel:
            continue  # geometry can only add; prune

        # R_a[k, i]: chain A's residual when A starts at i and B starts at k.
        r_a = _chain_first_scatter_residuals(geo_table, chain_a, chain_b)
        # R_b[i, k]: chain B's residual when B starts at k and A starts at i.
        r_b = _chain_first_scatter_residuals(geo_table, chain_b, chain_a)
        combined = r_a.T + r_b  # indexed [i, k]

        i_best, k_best = np.unravel_index(np.argmin(combined), combined.shape)
        sel = energy_resid + combined[i_best, k_best]

        if sel < best_sel:
            best_sel = float(sel)
            best = (float(combined[i_best, k_best]), int(i_best), int(k_best), chain_a.copy(), float(e_a), float(e_b))

    if best is None or not np.isfinite(best_sel):
        return LORResult(float("nan"), None, None, None, None, float("nan"), float("nan"))

    geo_resid, i_best, k_best, chain_a, e_a, e_b = best
    delta = positions[i_best] - positions[k_best]
    norm = np.linalg.norm(delta)
    axis = delta / norm if norm > 1e-9 else None
    score = geo_resid + _symmetric_energy_residual(e_a, e_b)
    return LORResult(float(score), axis, i_best, k_best, chain_a, e_a, e_b, selection_score=best_sel)


def annihilation_lor_score(positions, energies) -> float:
    """Feature-shaped wrapper: just the scalar score (lower = more
    annihilation-like; NaN when untestable)."""
    return annihilation_lor(positions, energies).score
