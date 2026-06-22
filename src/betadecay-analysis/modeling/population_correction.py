"""Population-level (efficiency / background-subtraction) correction.

The per-event classifier saturates at the ~0.89 F1 physics ceiling, but the
*science* deliverable is usually the NUMBER of 511 keV annihilation events, not a
per-event label. That number can be recovered nearly unbiased even at 0.89 F1,
because the per-event false positives and false negatives largely cancel in a
population estimate.

Method (standard cut-and-count with efficiency + background subtraction). For a
sample of ``N_total`` events of which ``S`` are true signal and ``N_total - S`` are
background, the classifier selects::

    N_pass = eps * S + beta * (N_total - S)

where the two rates are CLASS-CONDITIONAL and therefore prevalence-independent
(they transfer from the labelled simulation to data, unlike precision/purity):

    eps  = P(select | signal)     = recall            = TP / (TP + FN)
    beta = P(select | background) = false-positive rate = FP / (FP + TN)

Solving for the true signal count::

    S_hat = (N_pass - beta * N_total) / (eps - beta)

``eps``/``beta`` are calibrated on a labelled set (here: TRAIN); ``N_pass``/
``N_total`` are read off the set being measured (here: held-out EVAL). The naive
"purity" estimate ``N_pass * precision`` is NOT used: precision depends on the
sample prevalence, so it does not generalise.

Each multiplicity bucket has its own ``(eps, beta)`` and is corrected
independently; the total is the sum, with statistical errors added in quadrature.
A bucket with ``eps - beta`` below ``EPS_MIN`` carries no information about the
signal count (the selection is not signal-sensitive) and is reported as
unconstrained rather than divided through a near-zero denominator.
"""

import math

import wandb

# Minimum signal-sensitivity (eps - beta) for a bucket to constrain the count.
EPS_MIN = 0.05


def _binom_se(k: int, n: int) -> float:
    """Binomial standard error of the rate k/n (0 if n == 0)."""
    if n <= 0:
        return 0.0
    p = k / n
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def class_conditional_rates(counts: dict) -> dict:
    """eps (recall) and beta (FPR) with binomial SEs, from one bucket's counts."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n_sig = tp + fn
    n_bg = fp + tn
    return {
        "eps": (tp / n_sig) if n_sig else float("nan"),
        "eps_se": _binom_se(tp, n_sig),
        "beta": (fp / n_bg) if n_bg else float("nan"),
        "beta_se": _binom_se(fp, n_bg),
        "n_sig": n_sig,
        "n_bg": n_bg,
    }


def corrected_count(
    n_pass: int,
    n_total: int,
    eps: float,
    beta: float,
    eps_se: float,
    beta_se: float,
    n_pass_var: float | None = None,
) -> tuple[float, float]:
    """Background-subtracted signal-count estimate S_hat and its 1-sigma stat error.

    S_hat = (n_pass - beta * n_total) / (eps - beta). The error propagates the
    BINOMIAL selection noise on ``n_pass`` and the binomial calibration errors on
    ``eps``/``beta`` (treated independent; ``n_total`` is observed, treated as
    known). Conditional on the sample's true (S, B), the selection is two
    binomials, so Var(n_pass) = S*eps(1-eps) + B*beta(1-beta) -- NOT Poisson
    ``n_pass`` (which over-covers badly when eps is near 1). S, B are unknown, so
    plug in the estimate: S_hat and B_hat = n_total - S_hat. The partials simplify
    to dS/dn_pass = 1/d, dS/deps = -S/d, dS/dbeta = (S-n_total)/d with d = eps-beta.
    """
    denom = eps - beta
    s = (n_pass - beta * n_total) / denom
    if n_pass_var is None:
        s_hat = min(max(s, 0.0), float(n_total))  # clamp for a sane variance plug-in
        b_hat = n_total - s_hat
        n_pass_var = s_hat * eps * (1.0 - eps) + b_hat * beta * (1.0 - beta)
    d_npass = 1.0 / denom
    d_eps = -s / denom
    d_beta = (s - n_total) / denom
    var = d_npass**2 * n_pass_var + d_eps**2 * eps_se**2 + d_beta**2 * beta_se**2
    return s, math.sqrt(max(var, 0.0))


def population_estimate(train_by_bucket: dict, eval_by_bucket: dict) -> dict:
    """Per-bucket + total background-subtracted signal-count estimate.

    Rates are calibrated on ``train_by_bucket``; the selection (``N_pass``,
    ``N_total``) is read from ``eval_by_bucket``. Both are ``{bucket: counts}``
    where counts is the ``{"tp","fp","fn","tn",...}`` dict the evaluator produces.

    Returns a dict with per-bucket rows, the summed corrected estimate with its
    statistical error, the (sim-only) truth and bias for validation, and a pooled
    cross-check that ignores the bucket split.
    """
    rows = []
    s_total = 0.0
    var_total = 0.0
    true_total = 0
    true_constrained = 0       # truth the method can actually measure
    true_unconstrained = 0     # truth in unconstrained buckets (blind spot)
    n_pass_total = 0
    n_events_total = 0
    unconstrained = []

    for b in sorted(set(train_by_bucket) & set(eval_by_bucket)):
        r = class_conditional_rates(train_by_bucket[b])
        ev = eval_by_bucket[b]
        n_pass = ev["tp"] + ev["fp"]
        n_total = ev["tp"] + ev["fp"] + ev["fn"] + ev["tn"]
        true_s = ev["tp"] + ev["fn"]
        true_total += true_s
        n_pass_total += n_pass
        n_events_total += n_total

        eps, beta = r["eps"], r["beta"]
        constrained = (
            math.isfinite(eps) and math.isfinite(beta) and (eps - beta) >= EPS_MIN
        )
        row = {
            "bucket": b, "eps": eps, "eps_se": r["eps_se"], "beta": beta,
            "beta_se": r["beta_se"], "n_pass": n_pass, "n_total": n_total,
            "true_s": true_s, "constrained": constrained,
        }
        if constrained:
            s, se = corrected_count(n_pass, n_total, eps, beta, r["eps_se"], r["beta_se"])
            row["s_hat"], row["s_se"] = s, se
            s_total += s
            var_total += se**2
            true_constrained += true_s
        else:
            row["s_hat"], row["s_se"] = float("nan"), float("nan")
            unconstrained.append(b)
            true_unconstrained += true_s
        rows.append(row)

    # Pooled cross-check: one (eps, beta) over all buckets, ignoring the split.
    pooled_tr = {k: sum(train_by_bucket[b][k] for b in train_by_bucket)
                 for k in ("tp", "fp", "fn", "tn")}
    pr = class_conditional_rates(pooled_tr)
    pooled = None
    if math.isfinite(pr["eps"]) and math.isfinite(pr["beta"]) and (pr["eps"] - pr["beta"]) >= EPS_MIN:
        ps, pse = corrected_count(
            n_pass_total, n_events_total, pr["eps"], pr["beta"], pr["eps_se"], pr["beta_se"]
        )
        pooled = {"eps": pr["eps"], "beta": pr["beta"], "s_hat": ps, "s_se": pse}

    # Bias is measured against the truth the method can actually constrain;
    # unconstrained buckets are a separately-reported blind spot, not a bias.
    bias = (s_total - true_constrained) / true_constrained if true_constrained else float("nan")
    return {
        "rows": rows,
        "s_total": s_total,
        "s_total_se": math.sqrt(var_total),
        "true_total": true_total,
        "true_constrained": true_constrained,
        "true_unconstrained": true_unconstrained,
        "bias": bias,
        "n_pass_total": n_pass_total,
        "n_events_total": n_events_total,
        "unconstrained_buckets": unconstrained,
        "pooled": pooled,
    }


def log_population_estimate(train_by_bucket: dict, eval_by_bucket: dict, split_name: str = "eval") -> dict:
    """Compute, print, and W&B-log the population correction. Returns the estimate."""
    est = population_estimate(train_by_bucket, eval_by_bucket)

    print(f"\n{split_name} population correction "
          f"(eps/beta calibrated on TRAIN, applied to {split_name} selection):")
    print("  bucket   eps±se          beta±se         N_pass  N_total   S_hat±stat       (true)")
    for r in est["rows"]:
        if r["constrained"]:
            s_str = f"{r['s_hat']:7.1f} ± {r['s_se']:4.1f}"
        else:
            s_str = "  unconstrained "
        print(f"   n{r['bucket']}   {r['eps']:6.4f}±{r['eps_se']:.4f}  "
              f"{r['beta']:6.4f}±{r['beta_se']:.4f}  {r['n_pass']:6d}  {r['n_total']:7d}   "
              f"{s_str}   ({r['true_s']})")

    if est["unconstrained_buckets"]:
        print(f"  [unconstrained buckets {est['unconstrained_buckets']}: "
              f"eps-beta < {EPS_MIN}, selection not signal-sensitive; "
              f"true signal there = {est['true_unconstrained']} is a blind spot, "
              f"excluded from both the estimate and the bias]")

    print(f"  TOTAL corrected signal S = {est['s_total']:.1f} ± {est['s_total_se']:.1f} (stat)   "
          f"true(constrained) = {est['true_constrained']}   bias = {100*est['bias']:+.2f}%")
    print(f"  (raw selected N_pass = {est['n_pass_total']}, "
          f"uncorrected bias = {100*(est['n_pass_total']-est['true_total'])/est['true_total']:+.2f}%)")
    if est["pooled"]:
        p = est["pooled"]
        print(f"  pooled cross-check: eps={p['eps']:.4f} beta={p['beta']:.4f} "
              f"-> S = {p['s_hat']:.1f} ± {p['s_se']:.1f}")

    log = {
        f"{split_name}/population/S_corrected": est["s_total"],
        f"{split_name}/population/S_corrected_se": est["s_total_se"],
        f"{split_name}/population/S_true": est["true_total"],
        f"{split_name}/population/bias": est["bias"],
        f"{split_name}/population/N_pass": est["n_pass_total"],
    }
    if est["pooled"]:
        log[f"{split_name}/population/S_corrected_pooled"] = est["pooled"]["s_hat"]
    wandb.log(log)
    return est
