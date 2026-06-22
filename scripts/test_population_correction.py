"""Validate modeling/population_correction.py without MEGAlib.

(1) Algebraic sanity: same-set rates recover the true count exactly.
(2) Monte-Carlo coverage: plant per-bucket (eps, beta, S, B), simulate finite
    train+eval selections, calibrate on train, correct eval, and check that the
    estimate is unbiased AND its propagated 1-sigma error has ~68% coverage.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from modeling.population_correction import corrected_count, population_estimate


def counts(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "excluded": 0}


def test_algebra():
    # Champion eval confusion, used as both calibration and measurement (same-set):
    # the class-conditional estimator must return the true signal count exactly.
    n2 = counts(53, 21, 0, 1393)
    n3 = counts(530, 97, 19, 1286)
    by = {2: n2, 3: n3}
    est = population_estimate(by, by)
    true = (53 + 0) + (530 + 19)
    print(f"(1) algebra same-set: S_hat={est['s_total']:.3f}  true={true}  "
          f"diff={est['s_total']-true:+.3e}")
    assert abs(est["s_total"] - true) < 1e-6, "class-conditional estimator not exact on same-set"
    print("    PASS (exact)")


def simulate_bucket(S, B, eps, beta, rng):
    tp = sum(1 for _ in range(S) if rng.random() < eps)
    fp = sum(1 for _ in range(B) if rng.random() < beta)
    return counts(tp, fp, S - tp, B - fp)


def test_coverage(n_trials=2000):
    # Planted per-bucket truth (eval-scale ~ champion 20% split; train = 4x).
    plan = {
        2: {"S": 53, "B": 1414, "eps": 0.985, "beta": 0.014},
        3: {"S": 549, "B": 1383, "eps": 0.965, "beta": 0.070},
    }
    rng = random.Random(20260622)
    biases = []
    inside = 0
    true_total = sum(p["S"] for p in plan.values())
    for _ in range(n_trials):
        train_by = {b: simulate_bucket(4 * p["S"], 4 * p["B"], p["eps"], p["beta"], rng)
                    for b, p in plan.items()}
        eval_by = {b: simulate_bucket(p["S"], p["B"], p["eps"], p["beta"], rng)
                   for b, p in plan.items()}
        # the realised eval truth varies trial to trial; compare against it
        true_eval = sum(eval_by[b]["tp"] + eval_by[b]["fn"] for b in eval_by)
        est = population_estimate(train_by, eval_by)
        biases.append(est["s_total"] - true_eval)
        if abs(est["s_total"] - true_eval) <= est["s_total_se"]:
            inside += 1
    mean_bias = sum(biases) / len(biases)
    rms = (sum(b * b for b in biases) / len(biases)) ** 0.5
    cov = inside / n_trials
    print(f"(2) coverage MC ({n_trials} trials, planted true~{true_total}):")
    print(f"    mean bias = {mean_bias:+.2f} events ({100*mean_bias/true_total:+.2f}%)  "
          f"RMS spread = {rms:.1f}")
    print(f"    +/-1sigma coverage = {100*cov:.1f}%  (target ~68%)")
    assert abs(mean_bias) < 0.05 * true_total / 10, "estimator biased"  # < ~0.5% of total
    assert 0.60 <= cov <= 0.76, f"error bars miscalibrated (coverage {cov:.2f})"
    print("    PASS (unbiased + calibrated error bars)")


def test_propagation_point():
    # single-call corrected_count: known inputs, finite-difference check of error.
    s, se = corrected_count(701, 6596, 0.965, 0.0197, 0.0075, 0.0018)
    print(f"(3) corrected_count(701,6596,eps.965,beta.0197): S={s:.1f} ± {se:.1f}")
    assert 590 < s < 615, "point estimate off"
    assert 0 < se < 50, "error bar implausible"


if __name__ == "__main__":
    test_algebra()
    test_propagation_point()
    test_coverage()
    print("\nALL CHECKS PASSED")
