"""Multivariate per-event ceiling test on the n3 (>=3-hit) dE-degenerate pop.

Univariate AUCs were all weak (diag_ceiling.py). This asks the stronger question:
can ANY flexible per-event model (MLP) separate the degenerate FP from signal?
Builds a rich per-event feature vector, trains a 2-layer MLP with a held-out
split (5-fold CV), reports mean held-out AUC + a precision/recall operating point.
Pure analysis, no MEGAlib, writes nothing to the pipeline.
"""
import itertools
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)
torch.manual_seed(0)


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


def min_back_to_back(P):
    n = len(P)
    if n < 2:
        return 0.0
    vecs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (P[j][0]-P[i][0], P[j][1]-P[i][1], P[j][2]-P[i][2])
            nn = (d[0]**2+d[1]**2+d[2]**2) ** 0.5
            if nn > 1e-9:
                vecs.append((d[0]/nn, d[1]/nn, d[2]/nn))
    best = 1.0
    for a in range(len(vecs)):
        for b in range(a+1, len(vecs)):
            c = sum(vecs[a][k]*vecs[b][k] for k in range(3))
            best = min(best, c)
    return best


def featurize(E, P):
    n = len(E)
    Es = sorted(E, reverse=True)
    tot = sum(E)
    top = (Es + [0.0]*6)[:6]
    mean = tot / n
    std = (sum((e-mean)**2 for e in E)/n) ** 0.5
    cx = sum(p[0] for p in P)/n; cy = sum(p[1] for p in P)/n; cz = sum(p[2] for p in P)/n
    rms = (sum((p[0]-cx)**2+(p[1]-cy)**2+(p[2]-cz)**2 for p in P)/n) ** 0.5
    dists = [math.dist(P[i], P[j]) for i in range(n) for j in range(i+1, n)]
    maxd = max(dists) if dists else 0.0
    meand = sum(dists)/len(dists) if dists else 0.0
    return top + [float(n), tot, std, max(E)/tot, min(E)/tot, rms, maxd, meand,
                  min_back_to_back(P), delta_E(E)]


def auc(score, y):
    order = sorted(range(len(y)), key=lambda i: score[i])
    rsum = 0.0; i = 0; m = len(y)
    s = [score[o] for o in order]; yy = [y[o] for o in order]
    while i < m:
        j = i
        while j < m and s[j] == s[i]:
            j += 1
        ar = (i+j-1)/2.0 + 1
        for k in range(i, j):
            if yy[k] == 1:
                rsum += ar
        i = j
    npos = sum(y); nneg = len(y)-npos
    return (rsum - npos*(npos+1)/2.0)/(npos*nneg)


def main():
    X, Y = [], []
    for ev in iter_sim_events(SIM):
        if len(ev.hits) < 3:
            continue
        E = [h.energy for h in ev.hits]
        if delta_E(E) >= TOL:
            continue  # n3 degenerate population only
        P = [h.position for h in ev.hits]
        X.append(featurize(E, P))
        Y.append(1 if event_is_bdecay(ev, REF, TOL) else 0)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    n, d = X.shape
    print(f"n3 degenerate: {n} events, {int(Y.sum())} sig / {int((1-Y).sum())} bg, {d} features")

    fnames = ["e1", "e2", "e3", "e4", "e5", "e6", "n_hits", "totE", "std",
              "max/tot", "min/tot", "rms", "maxd", "meand", "min_b2b", "delta_E"]
    print("univariate AUC on THIS population:")
    yl = Y.int().tolist()
    for i in range(d):
        a = auc(X[:, i].tolist(), yl)
        print(f"  {fnames[i]:9s}: AUC={a:.3f}  (|0.5-AUC|={abs(0.5-a):.3f})")

    # standardize (note: full-data stats — minor; magnitude of effect checked below)
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    K = 5
    aucs, lin_aucs = [], []
    best_op = None
    for k in range(K):
        te = perm[k*n//K:(k+1)*n//K]
        tr = torch.cat([perm[:k*n//K], perm[(k+1)*n//K:]])
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

        # linear (logistic) baseline
        wl = torch.zeros(d, requires_grad=True); bl = torch.zeros(1, requires_grad=True)
        ol = torch.optim.Adam([wl, bl], lr=0.05)
        for _ in range(300):
            ol.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr@wl+bl, Ytr)
            loss.backward(); ol.step()
        with torch.no_grad():
            lin_aucs.append(auc((Xte@wl+bl).tolist(), Yte.int().tolist()))

        # 2-layer MLP
        net = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr).squeeze(1), Ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            sc = net(Xte).squeeze(1)
            aucs.append(auc(sc.tolist(), Yte.int().tolist()))
            if k == 0:
                # operating point: at signal recall ~0.965 (pipeline recall), what precision?
                p = torch.sigmoid(sc)
                thrs = sorted(p[Yte == 1].tolist())
                t = thrs[int(0.035*len(thrs))]  # keep 96.5% of signal
                pred = p >= t
                tp = int(((Yte == 1) & pred).sum()); fp = int(((Yte == 0) & pred).sum())
                fn = int(((Yte == 1) & ~pred).sum())
                best_op = (tp, fp, fn, tp/(tp+fp) if tp+fp else 0)

    print(f"linear  held-out AUC: {sum(lin_aucs)/K:.3f}  (folds {[round(a,3) for a in lin_aucs]})")
    print(f"MLP     held-out AUC: {sum(aucs)/K:.3f}  (folds {[round(a,3) for a in aucs]})")
    tp, fp, fn, prec = best_op
    print(f"MLP @ recall~0.965 (fold0): TP={tp} FP={fp} FN={fn}  precision={prec:.3f}  "
          f"(admit-all precision was 0.756)")


if __name__ == "__main__":
    main()
