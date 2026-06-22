"""TEST the population (efficiency-purity) correction before building it.

Builds a faithful delta_E-based per-bucket Bayesian classifier on the parser
(delta_E is production's dominant term), splits 80/20 like production, then tests:
  (a) naive purity recovery on the SAME set -> shows it is circular/trivial.
  (b) GENERALIZATION: estimate rates on TRAIN, correct the held-out EVAL count.
  (c) PREVALENCE INDEPENDENCE: resample eval to other signal fractions; show the
      purity estimator (p*N) becomes BIASED while the class-conditional estimator
      S=(N_pass - beta*N_total)/(eps-beta) stays unbiased.
  (d) bootstrap error bars.
No MEGAlib; pure analysis.
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


def delta_E(E):
    n = len(E)
    best = float("inf")
    for r in range(1, min(n + 1, 8)):
        for c in itertools.combinations(range(n), r):
            best = min(best, abs(sum(E[i] for i in c) - REF))
    return best


def bucket(nh):
    return 1 if nh == 1 else (2 if nh == 2 else 3)


def load():
    rows = []  # (label_int, bucket, delta_E)
    for ev in iter_sim_events(SIM):
        nh = len(ev.hits)
        if nh == 0:
            continue
        E = [h.energy for h in ev.hits]
        y = 1 if event_is_bdecay(ev, REF, TOL) else 0
        rows.append((y, bucket(nh), delta_E(E)))
    return rows


def hist_density(vals, edges):
    # returns normalized density per bin (+ Laplace smoothing)
    c = [0.0] * (len(edges) - 1)
    for v in vals:
        # clamp into range
        lo, hi = edges[0], edges[-1]
        vv = min(max(v, lo), hi - 1e-12)
        # binary search
        k = 0
        while k < len(edges) - 1 and not (edges[k] <= vv < edges[k + 1]):
            k += 1
        if k >= len(c):
            k = len(c) - 1
        c[k] += 1.0
    tot = sum(c) + 0.5 * len(c)
    return [(x + 0.5) / tot for x in c]


def log_edges(n_bins, lo=1e-3, hi=None, vals=None):
    if hi is None:
        hi = max(vals) if vals else 1.0
    hi = max(hi, lo * 10)
    a, b = math.log(lo), math.log(hi)
    return [math.exp(a + (b - a) * i / n_bins) for i in range(n_bins + 1)]


def train_eval_split(rows, seed=42, frac=0.8):
    g = torch.Generator().manual_seed(seed)
    by = {}
    for r in rows:
        by.setdefault((r[0], r[1]), []).append(r)
    train, ev = [], []
    for k, lst in by.items():
        idx = torch.randperm(len(lst), generator=g).tolist()
        nt = int(len(lst) * frac)
        for i in idx[:nt]:
            train.append(lst[i])
        for i in idx[nt:]:
            ev.append(lst[i])
    return train, ev


def fit(train, n_bins=35):
    models = {}
    for b in (1, 2, 3):
        sig = [r[2] for r in train if r[1] == b and r[0] == 1 and math.isfinite(r[2])]
        bg = [r[2] for r in train if r[1] == b and r[0] == 0 and math.isfinite(r[2])]
        n_sig = sum(1 for r in train if r[1] == b and r[0] == 1)
        n_bg = sum(1 for r in train if r[1] == b and r[0] == 0)
        if not sig or not bg:
            models[b] = None
            continue
        edges = log_edges(n_bins, vals=sig + bg)
        ds = hist_density(sig, edges)
        dg = hist_density(bg, edges)
        models[b] = (edges, ds, dg, n_sig, n_bg)
    return models


def predict(models, b, dE):
    m = models[b]
    if m is None or not math.isfinite(dE):
        return 0
    edges, ds, dg, n_sig, n_bg = m
    vv = min(max(dE, edges[0]), edges[-1] - 1e-12)
    k = 0
    while k < len(edges) - 1 and not (edges[k] <= vv < edges[k + 1]):
        k += 1
    if k >= len(ds):
        k = len(ds) - 1
    lr = ds[k] / dg[k]
    # MAP rule: posterior_sig >= posterior_bg  <=>  lr >= n_bg/n_sig
    return 1 if lr >= (n_bg / max(n_sig, 1)) else 0


def confusion(models, data):
    tp = fp = fn = tn = 0
    for y, b, dE in data:
        p = predict(models, b, dE)
        tp += (y == 1 and p == 1)
        fp += (y == 0 and p == 1)
        fn += (y == 1 and p == 0)
        tn += (y == 0 and p == 0)
    return tp, fp, fn, tn


def rates(tp, fp, fn, tn):
    eps = tp / (tp + fn) if tp + fn else 0      # recall / signal efficiency
    beta = fp / (fp + tn) if fp + tn else 0     # FPR / bg leakage
    p = tp / (tp + fp) if tp + fp else 0        # precision / purity
    return eps, beta, p


def correct_purity(N_pass, p):
    return N_pass * p


def correct_classcond(N_pass, N_total, eps, beta):
    if abs(eps - beta) < 1e-9:
        return float("nan")
    return (N_pass - beta * N_total) / (eps - beta)


def main():
    print("parsing + delta_E (one pass)...")
    rows = load()
    train, ev = train_eval_split(rows)
    models = fit(train)

    tp, fp, fn, tn = confusion(models, ev)
    eps, beta, p = rates(tp, fp, fn, tn)
    f1 = 2 * p * eps / (p + eps) if p + eps else 0
    print(f"\nEVAL confusion (proxy classifier): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  recall(eps)={eps:.4f}  FPR(beta)={beta:.4f}  precision(p)={p:.4f}  F1={f1:.4f}")

    # rates from TRAIN (the honest, generalization-valid source)
    tp_t, fp_t, fn_t, tn_t = confusion(models, train)
    eps_t, beta_t, p_t = rates(tp_t, fp_t, fn_t, tn_t)
    print(f"  TRAIN rates: eps={eps_t:.4f} beta={beta_t:.4f} p={p_t:.4f}")

    N_pass = tp + fp
    N_total = tp + fp + fn + tn
    S_true = tp + fn

    print("\n(a) SAME-SET purity 'recovery' (circular):")
    print(f"    N_pass*p(eval) = {correct_purity(N_pass, p):.1f} vs true {S_true} "
          f"-> exact by construction, proves nothing")

    print("\n(b) GENERALIZATION: TRAIN rates applied to held-out EVAL count")
    s_pur = correct_purity(N_pass, p_t)
    s_cc = correct_classcond(N_pass, N_total, eps_t, beta_t)
    print(f"    raw selected N_pass={N_pass}   true S_eval={S_true}")
    print(f"    purity est   (N_pass*p_train)              = {s_pur:.1f}  "
          f"bias {100*(s_pur-S_true)/S_true:+.2f}%")
    print(f"    class-cond   (N_pass-beta*N)/(eps-beta)    = {s_cc:.1f}  "
          f"bias {100*(s_cc-S_true)/S_true:+.2f}%")

    print("\n(c) PREVALENCE INDEPENDENCE (TRAIN rates fixed; resample eval prevalence)")
    print("    target_sig_frac  true_S  purity_est(bias%)   classcond_est(bias%)")
    ev_sig = [r for r in ev if r[0] == 1]
    ev_bg = [r for r in ev if r[0] == 0]
    g = torch.Generator().manual_seed(7)
    for target in (0.02, 0.0846, 0.20, 0.40):
        # build a resample with the target signal fraction (use all bg, subsample sig, or vice versa)
        nb = len(ev_bg)
        ns_target = int(target / (1 - target) * nb)
        if ns_target <= len(ev_sig):
            idx = torch.randperm(len(ev_sig), generator=g).tolist()[:ns_target]
            sub = [ev_sig[i] for i in idx] + ev_bg
        else:
            # need more signal than available -> subsample bg instead
            nb2 = int((1 - target) / target * len(ev_sig))
            idx = torch.randperm(len(ev_bg), generator=g).tolist()[:nb2]
            sub = ev_sig + [ev_bg[i] for i in idx]
        tp2, fp2, fn2, tn2 = confusion(models, sub)
        Np, Nt, St = tp2 + fp2, len(sub), tp2 + fn2
        sp = correct_purity(Np, p_t)
        sc = correct_classcond(Np, Nt, eps_t, beta_t)
        bp = 100 * (sp - St) / St if St else float("nan")
        bc = 100 * (sc - St) / St if St else float("nan")
        print(f"    {target:13.4f}  {St:6d}   {sp:7.1f} ({bp:+6.1f}%)   {sc:7.1f} ({bc:+6.1f}%)")

    print("\n(d) BOOTSTRAP error bar on the class-cond estimate (eval, 300 resamples)")
    g2 = torch.Generator().manual_seed(11)
    ests = []
    m = len(ev)
    for _ in range(300):
        idx = torch.randint(0, m, (m,), generator=g2).tolist()
        sub = [ev[i] for i in idx]
        tp2, fp2, fn2, tn2 = confusion(models, sub)
        Np, Nt = tp2 + fp2, len(sub)
        ests.append(correct_classcond(Np, Nt, eps_t, beta_t))
    ests.sort()
    lo, hi = ests[int(0.16 * len(ests))], ests[int(0.84 * len(ests))]
    mean = sum(ests) / len(ests)
    print(f"    S_est = {mean:.1f}  68% CI [{lo:.1f}, {hi:.1f}]  (true {S_true})")


if __name__ == "__main__":
    main()
