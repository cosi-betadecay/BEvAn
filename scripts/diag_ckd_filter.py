"""Postprocessing lever test: is the delta_E-achieving 511-subset a VALID
single-photon Compton path?

delta_E only checks that a subset SUMS to 511; it never checks the subset is a
kinematically consistent Compton sequence. Hypothesis: real annihilation 511
subsets are Compton-consistent (low CKD residual); coincidental-511 background
subsets are not. This is a self-contained (near-field) test -- no global
backprojection direction needed, so it works where far-field ARM fails.

For each >=3-hit delta_E-degenerate event: find the min-|sum-511| subset, compute
the best-ordering CKD residual (geometry vs Compton kinematics for an incident
511 keV photon), and compare signal vs background. Pure analysis, no MEGAlib.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
MEC2 = 511.0
TOL = 3.0 * (2.25 / 2.355)


def best_511_subset(E):
    n = len(E)
    best, bidx = float("inf"), None
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            d = abs(sum(E[i] for i in c) - REF)
            if d < best:
                best, bidx = d, c
    return best, bidx


def _unit(a, b):
    v = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    nn = (v[0]**2+v[1]**2+v[2]**2) ** 0.5
    return None if nn < 1e-9 else (v[0]/nn, v[1]/nn, v[2]/nn)


def ckd_residual(seq, E, P):
    """Mean squared (cos_geo - cos_kin) over internal scatters, incident 511 keV."""
    N = len(seq)
    if N < 3:
        return None  # no internal scatter to test
    Es = [E[h] for h in seq]
    pos = [P[h] for h in seq]
    W = [0.0]*N
    suffix = 0.0
    for i in range(N-1, -1, -1):
        W[i] = suffix / MEC2
        suffix += Es[i]
    dirs = [_unit(pos[i], pos[i+1]) for i in range(N-1)]
    resids = []
    for i in range(1, N-1):
        if dirs[i-1] is None or dirs[i] is None:
            continue
        w_in, w_out = W[i-1], W[i]
        if w_in < 1e-9 or w_out < 1e-9:
            continue
        cos_geo = sum(dirs[i-1][k]*dirs[i][k] for k in range(3))
        cos_kin = max(-1.0, min(1.0, 1.0 + 1.0/w_in - 1.0/w_out))
        resids.append((cos_geo - cos_kin)**2)
    if not resids:
        return None
    return sum(resids)/len(resids)


def best_ckd(subset, E, P):
    """Min CKD residual over orderings of `subset` (full perms if small)."""
    s = list(subset)
    if len(s) < 3:
        return None
    if len(s) <= 6:
        orders = itertools.permutations(s)
    else:
        # too big to fully enumerate: try energy-descending + a few rotations
        base = sorted(s, key=lambda h: E[h], reverse=True)
        orders = [tuple(base[k:]+base[:k]) for k in range(len(base))]
    best = float("inf")
    for o in orders:
        r = ckd_residual(o, E, P)
        if r is not None:
            best = min(best, r)
    return None if best == float("inf") else best


def auc(score, y):
    order = sorted(range(len(y)), key=lambda i: score[i])
    s = [score[o] for o in order]; yy = [y[o] for o in order]
    rsum = 0.0; i = 0; m = len(y)
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
    sig, bg = [], []          # (ckd_residual or None, n_subset)
    sig_none = bg_none = 0    # size-2 subsets: no CKD possible
    for ev in iter_sim_events(SIM):
        if len(ev.hits) < 3:
            continue
        E = [h.energy for h in ev.hits]
        dE, sub = best_511_subset(E)
        if dE >= TOL:
            continue
        P = [h.position for h in ev.hits]
        y = event_is_bdecay(ev, REF, TOL)
        r = best_ckd(sub, E, P)
        if r is None:
            if y: sig_none += 1
            else: bg_none += 1
            continue
        (sig if y else bg).append(r)

    print(f"n3 degenerate w/ CKD-testable 511-subset (size>=3): sig={len(sig)} bg={len(bg)}")
    print(f"  511-subset size<3 (energy-only, no CKD): sig={sig_none} bg={bg_none}")
    ys = [1]*len(sig) + [0]*len(bg)
    a = auc(sig + bg, ys)
    print(f"  CKD residual AUC (signal LOWER => a<0.5): {a:.3f}  (|0.5-AUC|={abs(0.5-a):.3f})")
    sig.sort(); bg.sort()
    print(f"  median CKD residual: sig={sig[len(sig)//2]:.4f}  bg={bg[len(bg)//2]:.4f}")
    print("\n  POSTPROCESSING FP-filter: keep-as-signal iff CKD residual <= thr")
    print("  thr      sig_kept(of %d)   bg_kept=FP(of %d)" % (len(sig), len(bg)))
    for thr in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        sk = sum(1 for r in sig if r <= thr)
        bk = sum(1 for r in bg if r <= thr)
        print(f"  {thr:5.3f}    {sk:5d} ({sk/len(sig):.3f})     {bk:4d} ({bk/len(bg):.3f})")


if __name__ == "__main__":
    main()
