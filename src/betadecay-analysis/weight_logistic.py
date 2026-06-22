"""Definitive weighting test (run on the MEGAlib VM).

Naive Bayes (the champion) is: predict signal iff  sum_i LLR_i + log(n_beta/n_bg) > 0,
i.e. every term's log-likelihood-ratio gets weight 1 and the threshold is the class
prior. This fits, per bucket, a LOGISTIC REGRESSION on those same per-term LLRs:

    P(signal) = sigmoid( b + sum_i c_i * LLR_i )

c_i are the provably-optimal linear weights and b the optimal threshold, found by
convex optimization on TRAIN (no CEM noise, no F1 hill-climbing). Champion is the
special case c_i = 1, b = log(n_beta/n_bg) -- a point in this model's space -- so
logistic can only match or beat it. Sanity check: the reconstructed champion F1
must equal the production 0.8935 (else the LLR reconstruction is wrong).

    python3 src/betadecay-analysis/weight_logistic.py
"""
import os

import hydra
import omegaconf
import torch
from dataset.datasets import _FEATURES, BUCKETS, Datasets
from dotenv import load_dotenv
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

import wandb

EPS = 1e-8


def bucket_llrs(model, data, bucket):
    """Per-event term LLR matrix X (n_valid, n_terms), labels y, and the champion
    intercept log(n_beta/n_bg). Mirrors Evaluator._evaluate_bucket's valid mask."""
    bd, bg = data["bdecay"][bucket], data["bg"][bucket]
    n_bd, n_bg = bd["delta_E"].numel(), bg["delta_E"].numel()
    if n_bd + n_bg == 0:
        return None
    feats = {f: torch.cat([bd[f], bg[f]]) for f in _FEATURES}
    y = torch.cat([torch.ones(n_bd), torch.zeros(n_bg)])
    used = ["delta_E"] + (["arm"] if bucket >= 2 else []) + (["anni"] if bucket >= 3 else [])
    valid = torch.ones(y.shape[0], dtype=torch.bool)
    for f in used:
        valid = valid & torch.isfinite(feats[f])
    feats = {f: v[valid] for f, v in feats.items()}
    y = y[valid]
    cols = []
    for spec in model["terms"]:
        if spec["kind"] == "1d":
            pb = lookup_density_values_1d(feats[spec["xfeat"]], spec["beta"], spec["x_bins"])
            pg = lookup_density_values_1d(feats[spec["xfeat"]], spec["bg"], spec["x_bins"])
        else:
            pb = lookup_density_values(feats[spec["xfeat"]], feats[spec["yfeat"]],
                                       spec["beta"], spec["x_bins"], spec["y_bins"])
            pg = lookup_density_values(feats[spec["xfeat"]], feats[spec["yfeat"]],
                                       spec["bg"], spec["x_bins"], spec["y_bins"])
        cols.append(torch.log(pb + EPS) - torch.log(pg + EPS))
    X = torch.stack(cols, dim=1) if cols else torch.zeros((y.shape[0], 0))
    prior = float(torch.log(torch.tensor(model["n_beta"] / max(model["n_bg"], 1))))
    return X, y, prior


def fit_logistic(X, y, iters=4000, lr=0.05):
    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def _conf(logit, y):
    pred = logit > 0
    return [int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()),
            int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())]


def _f1(c):
    tp, fp, fn, _ = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _best_f1(score, y):
    """Max pooled F1 over a swept global threshold, plus the confusion at that point.

    This is the part that makes the weighting test FAIR. Logistic regression fixes
    its threshold for likelihood (calibration), NOT for F1 -- so at its natural
    threshold it can sit at a worse-recall operating point even when its *combination*
    of the LLRs is as good as or better than naive. Sweeping the threshold for BOTH
    the naive score and the logistic score isolates the COMBINATION (the weights)
    from WHERE the cut sits. ``score`` is one pooled value per event across buckets."""
    s = score.tolist()
    order = sorted(range(len(s)), key=lambda i: s[i])
    ys = [int(y[i]) for i in order]
    P = sum(ys)
    tp, fp, fn, tn = P, len(ys) - P, 0, 0   # threshold below all -> predict all signal
    best, best_c = 0.0, (tp, fp, fn, tn)
    for k in range(len(ys)):                # raise threshold past event k (ascending score)
        if ys[k] == 1:
            tp -= 1
            fn += 1
        else:
            fp -= 1
            tn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best:
            best, best_c = f1, (tp, fp, fn, tn)
    return best, best_c


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    wandb.init(project=cfg.project_names[0])
    ref_energy = 511.0
    imager = FarFieldImager(geometry_file=cfg.setup.geo_file, n_phi=360, n_theta=180,
                            coordinate_system="spheric")
    imager.backproject_file(cfg.setup.tra_file)
    unit_vector = imager.peak_direction_cartesian()
    reader_sim, reader_tra = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)
    datasets = Datasets(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    data = datasets.compute_event_features(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    train, evaluate = datasets.split_dataset(data)
    trainer = Trainer(cfg).fit(train)  # champion densities

    tot_champ = [0, 0, 0, 0]
    tot_log = [0, 0, 0, 0]
    # Pooled per-event scores across buckets, so a single global threshold sweep can
    # find best-F1 for each combination. Both are centered so threshold 0 = natural cut:
    #   champion score = sum LLR + prior        (log posterior odds; champion cuts at 0)
    #   logistic score = b + sum c_i * LLR      (the fitted logit;   natural cut at 0)
    champ_scores, log_scores, all_y = [], [], []
    print("\nper-bucket optimal weights (logistic on the LLRs) vs naive Bayes:")
    for b in BUCKETS:
        m = trainer.models[b]
        tr = bucket_llrs(m, train, b)
        ev = bucket_llrs(m, evaluate, b)
        if tr is None or ev is None:
            continue
        Xtr, ytr, prior = tr
        Xev, yev, _ = ev
        cs = Xev.sum(dim=1) + prior                         # w=1, b=prior (= R >= tau)
        w, bb = fit_logistic(Xtr, ytr)
        ls = Xev @ w + bb
        champ = _conf(cs, yev)
        logi = _conf(ls, yev)
        champ_scores.append(cs)
        log_scores.append(ls)
        all_y.append(yev)
        for i in range(4):
            tot_champ[i] += champ[i]
            tot_log[i] += logi[i]
        terms = [s.get("yfeat") or s["xfeat"] for s in m["terms"]]
        print(f"  n{b}: champ w=[1.00]x{len(terms)} b={prior:+.2f}  ->  "
              f"logistic w={[round(x, 2) for x in w.tolist()]} b={float(bb):+.2f}  "
              f"(terms {terms})")

    cs = torch.cat(champ_scores)
    ls = torch.cat(log_scores)
    y = torch.cat(all_y)

    # (1) Natural-threshold confusions -- champion MUST be 0.8935 (LLR-reconstruction
    #     sanity). The logistic here sits at its likelihood-optimal cut, NOT its
    #     F1-optimal cut, so a drop here is the THRESHOLD moving, not the weighting.
    print("\n  -- at each model's natural threshold (logistic cut = likelihood, not F1) --")
    print(f"  champion (naive Bayes)  eval F1 = {_f1(tot_champ):.4f}   {tuple(tot_champ)}  "
          "<-- must be 0.8935 (sanity check)")
    print(f"  logistic (natural cut)  eval F1 = {_f1(tot_log):.4f}   {tuple(tot_log)}")

    # (2) THE FAIR TEST: best-F1 over a swept global threshold for BOTH combinations.
    #     This isolates the WEIGHTING (how the LLRs are combined) from the operating
    #     point. If logistic's better combination buys anything, it shows up HERE.
    champ_best, champ_bc = _best_f1(cs, y)
    log_best, log_bc = _best_f1(ls, y)
    delta = log_best - champ_best
    print("\n  -- best-F1 over a swept threshold (isolates WEIGHTING from the cut) --")
    print(f"  champion combination  best-F1 = {champ_best:.4f}   {champ_bc}")
    print(f"  logistic combination  best-F1 = {log_best:.4f}   {log_bc}")
    print(f"  delta = {delta:+.4f}  ->  "
          + ("LOGISTIC WEIGHTING HELPS (a better LLR combination exists)" if delta > 1e-4 else
             "no gain: naive-Bayes weighting is optimal -- R-weighting is exhausted"))
    wandb.log({"weight_logistic/champion_f1": _f1(tot_champ),
               "weight_logistic/logistic_natural_f1": _f1(tot_log),
               "weight_logistic/champion_best_f1": champ_best,
               "weight_logistic/logistic_best_f1": log_best})
    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    if os.getenv("WANDB_API_KEY") is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    main()
