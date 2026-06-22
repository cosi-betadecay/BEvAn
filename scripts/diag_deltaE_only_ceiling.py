"""Robustness check: does a PURE delta_E threshold (no density model, no buckets)
also beat production's 0.8935? And what is the truly-irreducible exactly-511
background floor?

If a one-parameter delta_E cut on held-out eval reaches ~0.99, the proxy's strength
is not an artifact of my density/bucket machinery -- it's that delta_E resolution
near 0 separates exactly-511 signal from spread coincidental background, and
production's geometry joints are diluting it. If it lands ~0.89, my density model
was doing something subtle and the claim is suspect.
"""
import itertools
import os
import sys

import torch

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


def main():
    rows = []  # (label, nh, dE)
    for ev in iter_sim_events(SIM):
        nh = len(ev.hits)
        if nh == 0:
            continue
        E = [h.energy for h in ev.hits]
        rows.append((1 if event_is_bdecay(ev, REF, TOL) else 0, nh, delta_E(E)))

    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(rows), generator=g).tolist()
    nt = int(0.8 * len(rows))
    train = [rows[i] for i in idx[:nt]]
    ev = [rows[i] for i in idx[nt:]]

    def f1_at(data, cut, min_hits=1):
        tp = fp = fn = tn = 0
        for y, nh, dE in data:
            pred = 1 if (dE <= cut and nh >= min_hits) else 0
            tp += y == 1 and pred == 1
            fp += y == 0 and pred == 1
            fn += y == 1 and pred == 0
            tn += y == 0 and pred == 0
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return f1, prec, rec, tp, fp, fn, tn

    print("PURE delta_E threshold on held-out EVAL (no density, no buckets):")
    print("  cut(keV)  min_hits  F1     prec    rec    TP   FP   FN")
    best = None
    for mh in (1, 2):
        for cut in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, TOL):
            f1, prec, rec, tp, fp, fn, tn = f1_at(ev, cut, mh)
            print(f"  {cut:7.3f}    {mh}      {f1:.4f}  {prec:.4f}  {rec:.4f}  {tp:4d} {fp:4d} {fn:3d}")
            if best is None or f1 > best[0]:
                best = (f1, cut, mh)
    print(f"  -> best eval F1 = {best[0]:.4f} at cut={best[1]} keV, min_hits={best[2]}")
    print("     (production shipped F1 = 0.8935 with delta_E + ARM + anni)")

    # Irreducible floor: background events with a TRULY exactly-511 subset (delta_E
    # below detector-resolution-free threshold). These are physically identical to
    # signal on energy and CANNOT be separated by any energy feature.
    print("\nIrreducible exactly-511 background floor (full dataset):")
    n_sig = sum(1 for y, _, _ in rows if y == 1)
    for eps in (0.001, 0.01, 0.05, 0.1, 0.2):
        bg_exact = sum(1 for y, _, dE in rows if y == 0 and dE <= eps)
        sig_exact = sum(1 for y, _, dE in rows if y == 1 and dE <= eps)
        # best-case precision if a perfect energy classifier admitted exactly these
        prec = sig_exact / (sig_exact + bg_exact) if sig_exact + bg_exact else 0
        print(f"  delta_E<={eps:.3f}: sig={sig_exact:4d}/{n_sig}  bg(irreducible FP)={bg_exact:4d}"
              f"  -> max precision {prec:.4f}")


if __name__ == "__main__":
    main()
