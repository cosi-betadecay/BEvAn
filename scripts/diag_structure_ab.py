"""De-risk before implementing: does an event-structure feature add INCREMENTAL
F1 over delta_E? Noised-parser A/B (energies + N(0,0.955 keV)/hit), per-bucket
histogram likelihood-ratio classifier, reusing the production density primitives.

Compares, on a held-out split:
  A: delta_E only             (1D per bucket)
  B: delta_E + total_E        (delta_E 1D  x  total_E 1D, independent factors)
  C: delta_E + (total_E,nhit) (delta_E 1D  x  2D joint of total_E,n_hits)

Buckets by n_hits (1 / 2 / >=3) as a proxy for the production feature-availability
buckets (arm/anni need the full pipeline). Absolute F1 will not match production;
the INCREMENTAL delta_E -> +structure comparison is the signal. No MEGAlib.
"""
import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    lookup_density_values,
    lookup_density_values_1d,
)
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)
SIGMA = 2.25 / 2.355
RNG = random.Random(0)


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


def collect():
    rows = []
    for ev in iter_sim_events(SIM):
        if not ev.hits:
            continue
        E = [h.energy + RNG.gauss(0, SIGMA) for h in ev.hits]
        y = 1 if event_is_bdecay(ev, REF, TOL) else 0
        nh = len(ev.hits)
        has_anni = 1.0 if any(ia.process == "ANNI" for ia in ev.ias) else 0.0
        rows.append({"y": y, "bucket": 1 if nh == 1 else (2 if nh == 2 else 3),
                     "dE": delta_E(E), "totE": max(sum(E), 1.0), "nh": float(nh),
                     "has_anni": has_anni})
    return rows


def split(rows, frac=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(rows), generator=g).tolist()
    k = int(frac * len(rows))
    return [rows[i] for i in idx[:k]], [rows[i] for i in idx[k:]]


def _t(rows, key):
    return torch.tensor([r[key] for r in rows], dtype=torch.float32)


def f1_for(mode, train, ev, nb=25):
    """Per-bucket histogram-LR classifier; returns pooled F1 over buckets."""
    tp = fp = fn = tn = 0
    for b in (1, 2, 3):
        tr = [r for r in train if r["bucket"] == b]
        te = [r for r in ev if r["bucket"] == b]
        trs = [r for r in tr if r["y"] == 1]
        trb = [r for r in tr if r["y"] == 0]
        if not trs or not trb or not te:
            # prior-only: predict majority (background) -> all negatives
            fn += sum(r["y"] for r in te)
            tn += sum(1 - r["y"] for r in te)
            continue
        prior = len(trb) / len(trs)  # MAP threshold on LR
        # delta_E 1D
        _, xb = build_density_matrix_1d(_t(tr, "dE"), n_bins_x=nb, spacing_x="log", log_x_floor=1e-3)
        sb, _ = build_density_matrix_1d(_t(trs, "dE"), x_bins=xb, smoothing=0.5)
        bb, _ = build_density_matrix_1d(_t(trb, "dE"), x_bins=xb, smoothing=0.5)
        lr = (lookup_density_values_1d(_t(te, "dE"), sb, xb) + 1e-8) / (
            lookup_density_values_1d(_t(te, "dE"), bb, xb) + 1e-8)
        if mode == "D":  # ORACLE annihilation-presence factor (truth has_ANNI, 2-bin)
            xb2 = torch.tensor([-0.5, 0.5, 1.5])
            sa, _ = build_density_matrix_1d(_t(trs, "has_anni"), x_bins=xb2, smoothing=0.5)
            ba, _ = build_density_matrix_1d(_t(trb, "has_anni"), x_bins=xb2, smoothing=0.5)
            lr = lr * (lookup_density_values_1d(_t(te, "has_anni"), sa, xb2) + 1e-8) / (
                lookup_density_values_1d(_t(te, "has_anni"), ba, xb2) + 1e-8)
        if mode in ("B", "C") and b >= 2:  # structure only meaningful for multi-hit
            if mode == "B":  # total_E 1D factor
                _, tb = build_density_matrix_1d(_t(tr, "totE"), n_bins_x=nb, spacing_x="log", log_x_floor=1.0)
                st, _ = build_density_matrix_1d(_t(trs, "totE"), x_bins=tb, smoothing=0.5)
                bt, _ = build_density_matrix_1d(_t(trb, "totE"), x_bins=tb, smoothing=0.5)
                lr = lr * (lookup_density_values_1d(_t(te, "totE"), st, tb) + 1e-8) / (
                    lookup_density_values_1d(_t(te, "totE"), bt, tb) + 1e-8)
            else:  # 2D joint (total_E, n_hits)
                _, tb, hbb = build_density_matrix(_t(tr, "totE"), _t(tr, "nh"), n_bins_x=nb, n_bins_y=12,
                                                  spacing_x="log", spacing_y="linear", log_x_floor=1.0)
                s2, _, _ = build_density_matrix(_t(trs, "totE"), _t(trs, "nh"), x_bins=tb, y_bins=hbb, smoothing=0.5)
                b2, _, _ = build_density_matrix(_t(trb, "totE"), _t(trb, "nh"), x_bins=tb, y_bins=hbb, smoothing=0.5)
                lr = lr * (lookup_density_values(_t(te, "totE"), _t(te, "nh"), s2, tb, hbb) + 1e-8) / (
                    lookup_density_values(_t(te, "totE"), _t(te, "nh"), b2, tb, hbb) + 1e-8)
        pred = lr >= prior
        yv = torch.tensor([r["y"] for r in te])
        tp += int(((yv == 1) & pred).sum()); fp += int(((yv == 0) & pred).sum())
        fn += int(((yv == 1) & ~pred).sum()); tn += int(((yv == 0) & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, tp, fp, fn, tn


def main():
    rows = collect()
    train, ev = split(rows)
    print(f"events: {len(rows)}  (noised parser)\n")
    for mode, desc in [("A", "delta_E only"), ("B", "delta_E + total_E"),
                       ("C", "delta_E + (total_E, n_hits) joint"),
                       ("D", "delta_E + has_ANNI [ORACLE/truth]")]:
        f1, tp, fp, fn, tn = f1_for(mode, train, ev)
        print(f"  [{mode}] {desc:38s}: F1={f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\nIncremental F1 (B-A, C-A) estimates whether event structure helps "
          "beyond delta_E. Positive -> worth wiring into the production model + VM A/B.")


if __name__ == "__main__":
    main()
