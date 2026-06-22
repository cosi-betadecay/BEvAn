"""Test the 'full beta+ event signature' hypothesis BEFORE implementing a feature.

Among delta_E-degenerate events (those with a ~511 subset -- where TP and FP both
live), does EVENT-LEVEL structure separate true signal from the background
contamination? Features tested:

  has_ANNI       : truth flag -- does the event contain a positron annihilation?
                   ORACLE upper bound on any annihilation/positron-presence feature.
  non_anni_E     : total energy NOT from annihilation-photon deposits (truth) --
                   ~ positron ionization + decay gammas. Oracle positron proxy.
  total_E        : total deposited energy (realizable)
  extra_E        : total_E - 511 (realizable positron+context proxy)
  n_hits, max_hit: realizable

CRUCIAL: the energy features live at the 100s-of-keV scale, so unlike delta_E
their separation should survive detector noising. We verify by re-measuring with
Gaussian noise (sigma=0.955 keV/hit, the production HPGe smearing) added.
Pure analysis; no MEGAlib.
"""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)
SIGMA = 2.25 / 2.355  # ~0.955 keV per-hit HPGe smearing
RNG = random.Random(0)


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


def anni_deposited_energy(ev):
    """Sum of hit energy from ANNI-photon secondaries (each hit counted once)."""
    sids = set()
    for ia in ev.ias:
        if ia.process == "ANNI":
            sids |= {o.ia_id for o in ev.ias if o.origin_id == ia.ia_id}
    if not sids:
        return 0.0
    tot = 0.0
    for h in ev.hits:
        if any(o in sids for o in h.origins):
            tot += h.energy
    return tot


def auc(score, y):
    order = sorted(range(len(y)), key=lambda i: score[i])
    s = [score[o] for o in order]; yy = [y[o] for o in order]
    rsum = 0.0; i = 0; m = len(y)
    while i < m:
        j = i
        while j < m and s[j] == s[i]:
            j += 1
        ar = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if yy[k] == 1:
                rsum += ar
        i = j
    npos = sum(y); nneg = len(y) - npos
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg) if npos and nneg else float("nan")


def mlp_auc(X, y, seed=0):
    """5-fold held-out AUC of a tiny MLP on feature matrix X (list of rows)."""
    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp_min(1e-6)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(y)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    aucs = []
    for k in range(5):
        te = perm[k * n // 5:(k + 1) * n // 5]
        tr = torch.cat([perm[:k * n // 5], perm[(k + 1) * n // 5:]])
        net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tr]).squeeze(1), yt[tr])
            loss.backward(); opt.step()
        with torch.no_grad():
            aucs.append(auc(net(Xt[te]).squeeze(1).tolist(), yt[te].int().tolist()))
    return sum(aucs) / len(aucs)


def collect(noise: bool):
    rows = []  # dict per degenerate event
    for ev in iter_sim_events(SIM):
        if not ev.hits:
            continue
        E = [h.energy + (RNG.gauss(0, SIGMA) if noise else 0.0) for h in ev.hits]
        if delta_E(E) >= TOL:
            continue
        y = 1 if event_is_bdecay(ev, REF, TOL) else 0
        total = sum(E)
        rows.append({
            "y": y,
            "total_E": total,
            "extra_E": total - REF,
            "n_hits": float(len(ev.hits)),
            "max_hit": max(E),
            "has_anni": 1.0 if any(ia.process == "ANNI" for ia in ev.ias) else 0.0,
            "non_anni_E": total - anni_deposited_energy(ev),
        })
    return rows


def report(rows, tag):
    y = [r["y"] for r in rows]
    print(f"\n[{tag}] degenerate events: {len(rows)}  signal={sum(y)}  bg={len(y)-sum(y)}")
    for f in ("has_anni", "non_anni_E", "total_E", "extra_E", "n_hits", "max_hit"):
        a = auc([r[f] for r in rows], y)
        kind = "ORACLE" if f in ("has_anni", "non_anni_E") else "real "
        print(f"  {kind} {f:11s}: AUC={a:.3f}  (|0.5-AUC|={abs(0.5-a):.3f})")
    realizable = ["total_E", "extra_E", "n_hits", "max_hit"]
    X = [[r[f] for f in realizable] for r in rows]
    print(f"  multivariate MLP (realizable {realizable}): held-out AUC={mlp_auc(X, y):.3f}")


def main():
    print("=== UN-NOISED (parser raw) ===")
    report(collect(noise=False), "un-noised")
    print("\n=== NOISED (sigma=0.955 keV/hit added; production-faithful) ===")
    report(collect(noise=True), "noised")
    print("\nIf AUCs are similar across the two blocks, the separation is "
          "noising-robust and the parser result transfers to production.")


if __name__ == "__main__":
    main()
