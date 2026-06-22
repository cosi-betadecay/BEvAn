"""Decisive check before extending annihilation_angle: can REALIZABLE hit-features
predict annihilation-presence (the oracle has_ANNI) on the FP population?

If a held-out model predicting has_ANNI from energies+positions scores high, the
single-photon+positron annihilation signature IS recoverable -> worth building into
annihilation_angle. If it's near chance, the oracle gain is truth-only and a real
feature can't reach it (it would just be another diluting total_E).

Includes positron-specific proxies: energy and max-hit OUTSIDE the best 511-subset
(the positron deposit is co-located energy that is NOT part of the 511 photon).
Noised energies (production-faithful). No MEGAlib.
"""
import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)
SIGMA = 2.25 / 2.355
EDGE = 511.0 * 2.0 / 3.0  # 511 keV Compton edge, 340.7
RNG = random.Random(0)


def best_511_subset(E):
    n = len(E)
    best, bidx = float("inf"), ()
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            d = abs(sum(E[i] for i in c) - REF)
            if d < best:
                best, bidx = d, c
    return best, set(bidx)


def min_back_to_back(P):
    n = len(P)
    if n < 2:
        return 1.0
    vecs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (P[j][0] - P[i][0], P[j][1] - P[i][1], P[j][2] - P[i][2])
            nn = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
            if nn > 1e-9:
                vecs.append((d[0] / nn, d[1] / nn, d[2] / nn))
    best = 1.0
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            best = min(best, sum(vecs[a][k] * vecs[b][k] for k in range(3)))
    return best


def featurize(E, P, sub):
    n = len(E)
    tot = sum(E)
    out = [E[i] for i in range(n) if i not in sub]  # hits outside the 511 subset
    e_out = sum(out)
    cx = sum(p[0] for p in P) / n; cy = sum(p[1] for p in P) / n; cz = sum(p[2] for p in P) / n
    rms = (sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2 for p in P) / n) ** 0.5
    return [
        tot, tot - REF, float(n), max(E), min(E),
        e_out,                                  # positron+context energy outside the 511 photon
        max(out) if out else 0.0,               # biggest non-photon deposit (positron candidate)
        float(len(out)),                        # # hits outside the photon
        sum(1 for e in E if e > EDGE),          # # hits above the 511 Compton edge
        min_back_to_back(P), rms,
    ]


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
    Xt = torch.tensor(X, dtype=torch.float32)
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp_min(1e-6)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(y); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g); aucs = []
    for k in range(5):
        te = perm[k * n // 5:(k + 1) * n // 5]
        tr = torch.cat([perm[:k * n // 5], perm[(k + 1) * n // 5:]])
        net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 24), torch.nn.ReLU(),
                                  torch.nn.Linear(24, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(350):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tr]).squeeze(1), yt[tr])
            loss.backward(); opt.step()
        with torch.no_grad():
            aucs.append(auc(net(Xt[te]).squeeze(1).tolist(), yt[te].int().tolist()))
    return sum(aucs) / len(aucs)


def main():
    X, y_label, y_anni = [], [], []
    for ev in iter_sim_events(SIM):
        if len(ev.hits) < 2:
            continue
        E = [h.energy + RNG.gauss(0, SIGMA) for h in ev.hits]
        dE, sub = best_511_subset(E)
        if dE >= TOL:
            continue  # FP population (has a ~511 subset)
        P = [h.position for h in ev.hits]
        X.append(featurize(E, P, sub))
        y_label.append(1 if event_is_bdecay(ev, REF, TOL) else 0)
        y_anni.append(1 if any(ia.process == "ANNI" for ia in ev.ias) else 0)

    n = len(X)
    print(f"FP-population (>=2 hit, 511-subset, noised): {n} events  "
          f"signal={sum(y_label)}  has_ANNI={sum(y_anni)}")
    fnames = ["total_E", "extra_E", "n_hits", "max_hit", "min_hit", "e_outside",
              "max_outside", "n_outside", "n_above_edge", "min_b2b", "rms"]
    print("\nrealizable feature -> predict has_ANNI (univariate AUC):")
    for i, fn in enumerate(fnames):
        print(f"  {fn:12s}: {auc([row[i] for row in X], y_anni):.3f}")
    print(f"\n  MLP realizable -> has_ANNI : held-out AUC = {mlp_auc(X, y_anni):.3f}  "
          "(oracle target; high => buildable)")
    print(f"  MLP realizable -> label    : held-out AUC = {mlp_auc(X, y_label):.3f}  "
          "(what actually helps the classifier)")


if __name__ == "__main__":
    main()
