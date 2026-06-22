"""Read-only ceiling diagnostic on data/Activation.sim via the pure-Python parser.

No MEGAlib. Reproduces the clean-511 label and a delta_E (min over subsets r<=7
of |sum E - 511|) and characterizes WHERE the residual errors live, and whether
any cheap per-event discriminant separates the delta_E-degenerate background
(would-be FP) from signal. Pure analysis; writes nothing to the pipeline.
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)  # ~2.866 keV, matches calculate_tolerance()
MEC2 = 511.0


def delta_E(energies):
    n = len(energies)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            s = sum(energies[i] for i in c)
            best = min(best, abs(s - REF))
    return best


def compton_phi(e_first, e_total):
    """cos scatter angle for a 511 photon: cos = 1 - mec2*(1/E_out - 1/E_in)."""
    e_out = e_total - e_first
    if e_out <= 0:
        return None
    cos = 1.0 - MEC2 * (1.0 / e_out - 1.0 / e_total)
    return max(-1.0, min(1.0, cos))


def auc(scores_pos, scores_neg):
    # higher score => more signal-like. Mann-Whitney.
    allv = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    allv.sort(key=lambda x: x[0])
    rsum = 0.0
    i = 0
    n = len(allv)
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if allv[k][1] == 1:
                rsum += avg_rank
        i = j
    npos, nneg = len(scores_pos), len(scores_neg)
    if npos == 0 or nneg == 0:
        return float("nan")
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    n_total = 0
    by_nhits = {}  # nhits_bucket -> [n_sig, n_bg]
    # delta_E-degenerate populations by approx bucket
    pop = {1: {"sig": [], "bg": []}, 2: {"sig": [], "bg": []}, 3: {"sig": [], "bg": []}}

    for ev in iter_sim_events(SIM):
        nh = len(ev.hits)
        if nh == 0:
            continue
        n_total += 1
        sig = event_is_bdecay(ev, REF, TOL)
        energies = [h.energy for h in ev.hits]
        positions = [h.position for h in ev.hits]
        dE = delta_E(energies)
        bkt = 1 if nh == 1 else (2 if nh == 2 else 3)
        by_nhits.setdefault(nh, [0, 0])
        by_nhits[nh][0 if sig else 1] += 1
        rec = {"dE": dE, "E": energies, "P": positions, "nh": nh}
        pop[bkt]["sig" if sig else "bg"].append(rec)

    print(f"total events (>=1 hit): {n_total}")
    sig_tot = sum(v[0] for v in by_nhits.values())
    print(f"signal: {sig_tot} ({100*sig_tot/n_total:.2f}%)  bg: {n_total-sig_tot}")
    print("\nn_hits :  signal   bg    sig_frac")
    for nh in sorted(by_nhits):
        s, b = by_nhits[nh]
        print(f"  {nh:3d}  : {s:6d} {b:6d}   {s/(s+b):.3f}" + ("  <-- " if nh <= 4 else ""))

    # Focus: delta_E-degenerate events (would be admitted by a delta_E classifier)
    print(f"\n=== delta_E-degenerate analysis (dE < {TOL:.3f} keV) ===")
    for bkt in (1, 2, 3):
        sig = [r for r in pop[bkt]["sig"] if r["dE"] < TOL]
        bg = [r for r in pop[bkt]["bg"] if r["dE"] < TOL]
        allsig = pop[bkt]["sig"]
        allbg = pop[bkt]["bg"]
        label = {1: "1-hit", 2: "2-hit", 3: ">=3-hit"}[bkt]
        print(f"\n[{label}] total sig={len(allsig)} bg={len(allbg)} | "
              f"dE-degenerate sig={len(sig)} bg={len(bg)}  "
              f"(precision ceiling if admit all = {len(sig)/(len(sig)+len(bg)):.3f})"
              if (len(sig)+len(bg)) else f"\n[{label}] empty")

    # 2-hit discriminants on the dE-degenerate population
    print("\n=== 2-hit dE-degenerate: candidate discriminants (AUC vs signal) ===")
    sig = [r for r in pop[2]["sig"] if r["dE"] < TOL]
    bg = [r for r in pop[2]["bg"] if r["dE"] < TOL]

    def feats(recs):
        phi, emin, emax, sep, asym = [], [], [], [], []
        for r in recs:
            E = r["E"]; P = r["P"]
            tot = sum(E)
            # 511 may be in singleton or in the 2-hit sum; use first=larger
            e1, e2 = max(E), min(E)
            c = compton_phi(e2, tot)  # photon scatters depositing e2 first? try both
            c2 = compton_phi(e1, tot)
            cc = c if c is not None else (c2 if c2 is not None else 0.0)
            phi.append(cc)
            emin.append(e2); emax.append(e1)
            d = math.dist(P[0], P[1]) if len(P) >= 2 else 0.0
            sep.append(d)
            asym.append(abs(e1 - e2) / (tot + 1e-9))
        return {"phi": phi, "emin": emin, "emax": emax, "sep": sep, "asym": asym}

    fs, fb = feats(sig), feats(bg)
    for k in ("phi", "emin", "emax", "sep", "asym"):
        a = auc(fs[k], fb[k])
        # report both orientations
        print(f"  {k:5s}: AUC={a:.3f} (|0.5-AUC|={abs(0.5-a):.3f})")
    print(f"  [n sig={len(sig)} bg={len(bg)}]")

    # Compton-edge cut: a real 511 photon's single Compton hit <= 340.67 keV.
    COMPTON_EDGE = 511.0 * (2.0 / 3.0)  # 340.67 keV
    print(f"\n=== 2-hit dE-degenerate: Compton-edge cut (max hit <= {COMPTON_EDGE:.1f} keV) ===")
    sig_pass = sum(1 for r in sig if max(r["E"]) <= COMPTON_EDGE)
    bg_pass = sum(1 for r in bg if max(r["E"]) <= COMPTON_EDGE)
    print(f"  signal kept: {sig_pass}/{len(sig)} ({sig_pass/len(sig):.3f})  "
          f"bg kept(=FP): {bg_pass}/{len(bg)} ({bg_pass/len(bg):.3f})")
    print(f"  -> FP removed: {len(bg)-bg_pass}, TP lost: {len(sig)-sig_pass}")
    # threshold scan
    print("  thr(keV)  sig_kept  bg_kept(FP)")
    for thr in (320, 340.67, 360, 380, 400, 450, 500):
        sk = sum(1 for r in sig if max(r["E"]) <= thr)
        bk = sum(1 for r in bg if max(r["E"]) <= thr)
        print(f"   {thr:6.1f}   {sk:4d}/{len(sig)}   {bk:3d}/{len(bg)}")

    # >=3-hit dE-degenerate: bigger FP bucket. Compton-edge of the LEADING hit
    # (max single deposit) is still bounded for a real 511 photon's first scatter.
    print("\n=== >=3-hit dE-degenerate: max-single-hit vs 511 Compton edge ===")
    sig3 = [r for r in pop[3]["sig"] if r["dE"] < TOL]
    bg3 = [r for r in pop[3]["bg"] if r["dE"] < TOL]
    print(f"  n sig={len(sig3)} bg={len(bg3)}")
    a = auc([max(r["E"]) for r in sig3], [max(r["E"]) for r in bg3])
    print(f"  max-hit-energy AUC={a:.3f}")
    print("  thr(keV)  sig_kept  bg_kept(FP)")
    for thr in (340.67, 380, 420, 460, 511):
        sk = sum(1 for r in sig3 if max(r["E"]) <= thr)
        bk = sum(1 for r in bg3 if max(r["E"]) <= thr)
        print(f"   {thr:6.1f}   {sk:5d}/{len(sig3)}   {bk:4d}/{len(bg3)}")

    # >=3-hit dE-degenerate: geometry / multiplicity discriminants
    def spread(r):
        P = r["P"]
        cx = sum(p[0] for p in P) / len(P)
        cy = sum(p[1] for p in P) / len(P)
        cz = sum(p[2] for p in P) / len(P)
        return (sum((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 for p in P) / len(P)) ** 0.5
    def maxpair(r):
        P = r["P"]
        return max(math.dist(P[i], P[j]) for i in range(len(P)) for j in range(i+1, len(P)))
    print("  geometry/multiplicity AUC (sig vs bg, >=3 dE-degenerate):")
    for name, fn in (("rms_spread", spread), ("max_pair_dist", maxpair),
                     ("n_hits", lambda r: r["nh"]), ("total_E", lambda r: sum(r["E"]))):
        a = auc([fn(r) for r in sig3], [fn(r) for r in bg3])
        print(f"    {name:14s}: AUC={a:.3f} (|0.5-AUC|={abs(0.5-a):.3f})")


if __name__ == "__main__":
    main()
