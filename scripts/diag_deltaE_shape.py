"""Is the delta_E within-window separation real & untapped, or a label artifact?

For >=3-hit events: histogram delta_E (min over subsets of |sumE-511|) for signal
vs background, both inside [0, TOL] and over the full range; report AUC of delta_E
over the FULL n3 population (not cut), and the delta_E of the ANNI subset vs the
global-min subset for signal (to see if the global min is a *different* coincidental
subset than the labeled annihilation one).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


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
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    sig_dE, bg_dE = [], []
    full_dE, full_y = [], []
    for ev in iter_sim_events(SIM):
        if len(ev.hits) < 3:
            continue
        E = [h.energy for h in ev.hits]
        dE = delta_E(E)
        y = 1 if event_is_bdecay(ev, REF, TOL) else 0
        full_dE.append(dE); full_y.append(y)
        if dE < TOL:
            (sig_dE if y else bg_dE).append(dE)

    print(f"FULL >=3-hit population: {sum(full_y)} sig / {len(full_y)-sum(full_y)} bg")
    print(f"  delta_E AUC over FULL n3 (signal lower?): {auc(full_dE, full_y):.3f}")
    # cap delta_E at a few values to see where the AUC comes from
    for cap in (TOL, 5, 10, 20, 50):
        capped = [min(d, cap) for d in full_dE]
        print(f"  delta_E capped at {cap:5.2f} keV -> AUC {auc(capped, full_y):.3f}")

    print(f"\nWithin window [0,{TOL:.3f}]: sig={len(sig_dE)} bg={len(bg_dE)}")
    edges = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, TOL]
    print("  bin(keV)        sig     bg    bg_frac")
    for a, b in zip(edges[:-1], edges[1:]):
        s = sum(1 for d in sig_dE if a <= d < b)
        g = sum(1 for d in bg_dE if a <= d < b)
        tot = s + g
        print(f"  [{a:.2f},{b:.2f}) : {s:6d} {g:6d}   {g/tot:.3f}" if tot else
              f"  [{a:.2f},{b:.2f}) :   empty")
    # median delta_E each class
    sig_dE.sort(); bg_dE.sort()
    print(f"  median delta_E  sig={sig_dE[len(sig_dE)//2]:.3f}  bg={bg_dE[len(bg_dE)//2]:.3f}")
    print(f"  frac with delta_E<0.2 keV: sig={sum(1 for d in sig_dE if d<0.2)/len(sig_dE):.3f}"
          f"  bg={sum(1 for d in bg_dE if d<0.2)/len(bg_dE):.3f}")


if __name__ == "__main__":
    main()
