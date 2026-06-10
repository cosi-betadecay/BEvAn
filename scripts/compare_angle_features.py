"""Three-way comparison: legacy annihilation_angle vs blind LOR score vs
LOR-preprocessed annihilation angle.

Methodology (the truth-harness one, plus the V4 lesson):
- full pass over data/Activation.sim via the pure-Python text reader;
- the legacy feature is computed through the VERBATIM production path
  (event_data_processing + physics_factors.annihilation_angle, champion
  state incl. the V3 soft energy penalty) using a shim event object;
- marginal class overlap (TV/BC, test_overlaps protocol) per feature;
- conditional-on-deltaE TV: count-weighted average TV inside deltaE
  quantile bins — approximates P(feature|dE), the quantity the classifier
  actually consumes (V4: marginal TV had four MCC inversions);
- Spearman correlations between features on jointly-finite events.

The preprocessed variant: run the LOR reconstruction, then compute the
legacy-style "most anti-parallel pair" min-cosine over the CLEANED vector
pool — the LOR line (photon A's travel direction) plus each chain's ordered
travel legs — instead of all C(n,2)-choose-2 inter-hit vector pairs.

Usage: .venv/bin/python scripts/compare_angle_features.py [--cache PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "betadecay-analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "tests_specialized"))

import torch  # noqa: E402

from mathematics.calculations import calculate_tolerance  # noqa: E402
from physics.annihilation_lor import (  # noqa: E402
    ELECTRON_MASS_KEV,
    PHOTON_KEV,
    _geo_residual_table,
    annihilation_lor,
)
from physics.event_processing import event_data_processing  # noqa: E402
from physics.physics_factors import annihilation_angle, delta_E  # noqa: E402
from utils.sim_text_reader import event_is_bdecay, iter_sim_events  # noqa: E402

SIM_FILE = "data/Activation.sim"


# ---------------------------------------------------------------------------
# Shim: feed the text-reader event through the verbatim production pipeline
# ---------------------------------------------------------------------------


class _Pos:
    __slots__ = ("_p",)

    def __init__(self, p):
        self._p = p

    def X(self):
        return self._p[0]

    def Y(self):
        return self._p[1]

    def Z(self):
        return self._p[2]


class _HT:
    __slots__ = ("_pos", "_e")

    def __init__(self, pos, e):
        self._pos = _Pos(pos)
        self._e = e

    def GetEnergy(self):
        return self._e

    def GetPosition(self):
        return self._pos


class _EventShim:
    __slots__ = ("_hts",)

    def __init__(self, hits):
        self._hts = [_HT(h.position, h.energy) for h in hits]

    def GetNHTs(self):
        return len(self._hts)

    def GetHTAt(self, i):
        return self._hts[i]


# ---------------------------------------------------------------------------
# LOR-preprocessed angle
# ---------------------------------------------------------------------------


def _chain_order(positions, energies, chain_mask, first, other_first, geo_table):
    """Ordered hit indices of one chain, replicating the module's walk:
    second = first-scatter argmin given the LOR incoming direction, then a
    greedy energy-ladder consistency walk (as in _internal_chain_residual)."""
    chain_idx = list(np.flatnonzero(chain_mask))
    if len(chain_idx) == 1:
        return chain_idx
    candidates = [j for j in chain_idx if j != first]
    second = min(candidates, key=lambda j: geo_table[other_first, first, j])

    order = [first, second]
    e_in = PHOTON_KEV - energies[first]
    prev, cur = first, second
    remaining = [j for j in chain_idx if j not in (first, second)]
    while remaining:
        e_out = e_in - energies[cur]
        if e_in <= 1e-6 or e_out <= 1e-6:
            order.extend(remaining)  # ladder exhausted; append arbitrarily
            break
        cos_kin = min(1.0, max(-1.0, 1.0 - ELECTRON_MASS_KEV * (1.0 / e_out - 1.0 / e_in)))
        d_in = positions[cur] - positions[prev]
        n_in = np.linalg.norm(d_in)
        best, best_j = None, None
        for j in remaining:
            d_out = positions[j] - positions[cur]
            n_out = np.linalg.norm(d_out)
            if n_in < 1e-9 or n_out < 1e-9:
                r = 0.0
            else:
                r = (float(np.dot(d_in, d_out) / (n_in * n_out)) - cos_kin) ** 2
            if best is None or r < best:
                best, best_j = r, j
        order.append(best_j)
        remaining.remove(best_j)
        e_in = e_out
        prev, cur = cur, best_j
    return order


def lor_preprocessed_angle(positions, energies):
    """Legacy-style min cosine over the LOR-cleaned vector pool.

    Pool = the LOR direction (photon A's travel direction, firstB -> firstA)
    + each chain's ordered travel legs. Returns the minimum pairwise cosine
    (most anti-parallel pair), NaN when fewer than two pool vectors exist
    (2-hit events: the pool is just the LOR line) or the LOR is untestable.
    """
    res = annihilation_lor(positions, energies)
    if res.axis is None:
        return float("nan")
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    geo_table = _geo_residual_table(positions, energies)

    chain_a = res.chain_a
    chain_b = ~chain_a
    vectors = [res.axis]  # photon A travel direction = unit(firstA - firstB)
    for mask, first, other in ((chain_a, res.first_a, res.first_b), (chain_b, res.first_b, res.first_a)):
        order = _chain_order(positions, energies, mask, first, other, geo_table)
        for a, b in zip(order[:-1], order[1:]):
            d = positions[b] - positions[a]
            n = np.linalg.norm(d)
            if n > 1e-9:
                vectors.append(d / n)
    if len(vectors) < 2:
        return float("nan")
    v = np.stack(vectors)
    cos = v @ v.T
    iu = np.triu_indices(len(vectors), k=1)
    return float(cos[iu].min())


# ---------------------------------------------------------------------------
# Feature computation (cached)
# ---------------------------------------------------------------------------


def compute(cache_path):
    tol = calculate_tolerance()
    rows = {"label": [], "n_hits": [], "dE": [], "legacy": [], "lor": [], "pre": []}
    t0 = time.time()
    for idx, ev in enumerate(iter_sim_events(SIM_FILE)):
        if not ev.hits:
            continue
        pos = np.array([h.position for h in ev.hits], dtype=np.float64)
        ene = np.array([h.energy for h in ev.hits], dtype=np.float64)

        shim = _EventShim(ev.hits)
        energies_t, positions_t, sizes_t = event_data_processing(shim)
        with torch.no_grad():
            legacy = float(
                annihilation_angle(positions_t, n_hits=len(ev.hits), sizes=sizes_t, energies=energies_t).item()
            )
            de = float(delta_E(energies_t, sizes=sizes_t).item())

        rows["label"].append(event_is_bdecay(ev, 511.0, tol))
        rows["n_hits"].append(len(ev.hits))
        rows["dE"].append(de)
        rows["legacy"].append(legacy)
        rows["lor"].append(annihilation_lor(pos, ene).score)
        rows["pre"].append(lor_preprocessed_angle(pos, ene))
        if idx % 10000 == 0 and idx:
            print(f"  ...{idx} events ({time.time()-t0:.0f}s)", flush=True)

    arrays = {k: np.array(v) for k, v in rows.items()}
    np.savez(cache_path, **arrays)
    print(f"computed {arrays['label'].size} events in {time.time()-t0:.0f}s -> {cache_path}")
    return arrays


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _tv(a, b, edges):
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return np.nan
    return 0.5 * float(np.abs(pa / pa.sum() - pb / pb.sum()).sum())


def _bc(a, b, edges):
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return np.nan
    return float(np.sqrt((pa / pa.sum()) * (pb / pb.sum())).sum())


def _transform(name, x):
    return np.log10(x + 1e-3) if name == "lor" else x  # angles are cosines, lor is a residual


def _edges(values):
    lo, hi = float(values.min()), float(values.max())
    return np.linspace(lo, hi if hi > lo else lo + 1e-6, 81)


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(x, y):
    if x.size < 3:
        return np.nan
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def report(t):
    L = t["label"].astype(bool)
    dE = t["dE"]
    feats = ["legacy", "lor", "pre"]
    names = {"legacy": "legacy annihilation_angle", "lor": "blind LOR score", "pre": "LOR-preprocessed angle"}

    print("\n=== coverage (finite values) ===")
    for f in feats:
        fin = np.isfinite(t[f])
        print(
            f"{names[f]:28s} signal {fin[L].sum():5d}/{L.sum()} ({fin[L].mean():.1%})   "
            f"bg {fin[~L].sum():6d}/{(~L).sum()} ({fin[~L].mean():.1%})"
        )

    print("\n=== marginal class overlap ===")
    for f in feats:
        fin = np.isfinite(t[f])
        a = _transform(f, t[f][fin & L])
        b = _transform(f, t[f][fin & ~L])
        edges = _edges(np.concatenate([a, b]))
        print(f"{names[f]:28s} TV = {_tv(a, b, edges):.3f}   BC = {_bc(a, b, edges):.3f}")

    print("\n=== conditional-on-deltaE TV (count-weighted over 15 dE quantile bins, >=30/class per bin) ===")
    fin_de = np.isfinite(dE)
    qs = np.quantile(dE[fin_de], np.linspace(0, 1, 16))
    qs[0], qs[-1] = -np.inf, np.inf
    for f in feats:
        total_w, acc = 0, 0.0
        for lo, hi in zip(qs[:-1], qs[1:]):
            band = fin_de & (dE >= lo) & (dE < hi) & np.isfinite(t[f])
            a = _transform(f, t[f][band & L])
            b = _transform(f, t[f][band & ~L])
            if a.size < 30 or b.size < 30:
                continue
            edges = _edges(np.concatenate([a, b]))
            w = band.sum()
            acc += w * _tv(a, b, edges)
            total_w += w
        print(f"{names[f]:28s} cond-TV = {acc/total_w:.3f}   (events covered: {total_w})")

    print("\n=== Spearman correlations (jointly finite, per class) ===")
    pairs = [("legacy", "lor"), ("legacy", "pre"), ("lor", "pre"), ("legacy", "dE"), ("lor", "dE"), ("pre", "dE")]
    for x, y in pairs:
        both = np.isfinite(t[x]) & np.isfinite(t[y] if y != "dE" else dE)
        ty = t[y] if y != "dE" else dE
        rs = _spearman(t[x][both & L], ty[both & L])
        rb = _spearman(t[x][both & ~L], ty[both & ~L])
        print(f"{x:7s} vs {y:7s}: signal rho = {rs:+.3f}   bg rho = {rb:+.3f}   (n={both.sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/angle_feature_comparison.npz")
    args = ap.parse_args()
    cache = Path(args.cache)
    if cache.exists():
        print(f"loading cache {cache}")
        t = dict(np.load(cache))
    else:
        t = compute(cache)
    report(t)
