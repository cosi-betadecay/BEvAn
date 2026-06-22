"""Postprocessing lever: cross-reference .sim events against the Revan .tra
reconstruction, joined on event ID.

delta_E is an oracle (min over all 2^n subsets of |sumE-511|) -> fires even on
random hits that coincidentally sum to 511. Revan can't cherry-pick: it reconstructs
ONE physical Compton/photo sequence. Hypothesis: for the delta_E-degenerate
population, a real annihilation reconstructs in Revan at total energy ~511 (or is a
clean PH/CO), while coincidental-511 background reconstructs elsewhere or is dropped.

Measures, on the delta_E-degenerate population: presence-in-.tra, reconstructed
total energy, and event type, as signal/background discriminants + a postprocessing
FP-filter operating point. Pure analysis, no MEGAlib.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
TRA = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.tra")
REF = 511.0
TOL = 3.0 * (2.25 / 2.355)


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


def parse_tra(path):
    """id -> (event_type, reconstructed_total_energy_keV)."""
    out = {}
    cur_id = None
    cur_et = None
    cur_pe = None
    cur_ch_sum = 0.0
    has_ch = False

    def flush():
        if cur_id is None:
            return
        if cur_et == "PH" and cur_pe is not None:
            E = cur_pe
        elif has_ch:
            E = cur_ch_sum
        else:
            E = None
        out[cur_id] = (cur_et, E)

    with open(path) as fh:
        for line in fh:
            if line.startswith("SE"):
                flush()
                cur_id, cur_et, cur_pe, cur_ch_sum, has_ch = None, None, None, 0.0, False
            elif line.startswith("ID "):
                cur_id = int(line.split()[1])
            elif line.startswith("ET "):
                cur_et = line.split()[1]
            elif line.startswith("PE "):
                cur_pe = float(line.split()[1])
            elif line.startswith("CH "):
                # CH idx x y z E t ...
                parts = line.split()
                if len(parts) >= 6:
                    cur_ch_sum += float(parts[5])
                    has_ch = True
    flush()
    return out


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


def main():
    tra = parse_tra(TRA)
    print(f".tra events parsed: {len(tra)}")

    # degenerate population (all multiplicities) cross-referenced
    sig_recon, bg_recon = [], []       # |E_recon - 511| for events present in .tra
    present = {"sig": [0, 0], "bg": [0, 0]}  # [present, absent]
    type_counts = {"sig": {}, "bg": {}}
    for ev in iter_sim_events(SIM):
        if not ev.hits:
            continue
        E = [h.energy for h in ev.hits]
        if delta_E(E) >= TOL:
            continue
        y = event_is_bdecay(ev, REF, TOL)
        key = "sig" if y else "bg"
        rec = tra.get(ev.event_id)
        if rec is None:
            present[key][1] += 1
            continue
        present[key][0] += 1
        et, Er = rec
        type_counts[key][et] = type_counts[key].get(et, 0) + 1
        if Er is not None:
            (sig_recon if y else bg_recon).append(abs(Er - REF))

    sp, sa = present["sig"]; bp, ba = present["bg"]
    print("\nDelta_E-degenerate population cross-referenced to .tra:")
    print(f"  signal: present={sp} absent={sa}  (present frac {sp/(sp+sa):.3f})")
    print(f"  bg    : present={bp} absent={ba}  (present frac {bp/(bp+ba):.3f})")
    print(f"  event types: sig={type_counts['sig']}  bg={type_counts['bg']}")

    print(f"\n|E_recon - 511| among .tra-present (sig={len(sig_recon)} bg={len(bg_recon)}):")
    ys = [1]*len(sig_recon) + [0]*len(bg_recon)
    a = auc(sig_recon + bg_recon, ys)
    print(f"  AUC (signal closer to 511 => a<0.5): {a:.3f}  (|0.5-AUC|={abs(0.5-a):.3f})")
    sr = sorted(sig_recon); br = sorted(bg_recon)
    print(f"  median |E_recon-511|: sig={sr[len(sr)//2]:.2f}  bg={br[len(br)//2]:.2f} keV")

    # POSTPROCESSING filter: signal iff present in .tra AND |E_recon-511| <= window.
    # Evaluate against the FULL degenerate population (absent => rejected).
    print(f"\nPOSTPROCESSING filter on degenerate pop "
          f"(total sig={sp+sa} bg={bp+ba}; admit-all precision={ (sp+sa)/(sp+sa+bp+ba):.3f}):")
    print("  window(keV)  TP_kept   FP_kept   precision   recall")
    tot_sig = sp + sa; tot_bg = bp + ba
    # rebuild per-event records for filtering
    recs = []
    for ev in iter_sim_events(SIM):
        if not ev.hits:
            continue
        E = [h.energy for h in ev.hits]
        if delta_E(E) >= TOL:
            continue
        y = event_is_bdecay(ev, REF, TOL)
        rec = tra.get(ev.event_id)
        Er = rec[1] if rec is not None else None
        recs.append((y, Er))
    for w in (2.0, 3.0, 5.0, 8.0, 12.0, 1e9):
        tp = sum(1 for y, Er in recs if y and Er is not None and abs(Er-REF) <= w)
        fp = sum(1 for y, Er in recs if (not y) and Er is not None and abs(Er-REF) <= w)
        prec = tp/(tp+fp) if tp+fp else 0
        rec_ = tp/tot_sig
        print(f"  {w:9.1f}   {tp:5d}/{tot_sig}  {fp:4d}/{tot_bg}   {prec:.3f}      {rec_:.3f}")


if __name__ == "__main__":
    main()
