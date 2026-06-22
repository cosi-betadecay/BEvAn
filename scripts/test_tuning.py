"""Validate pipeline/tuning.py on the Mac (no MEGAlib) with synthetic features.

(1) CEM converges on a known quadratic.
(2) lean_fit_eval runs through the REAL _fit_bucket/_evaluate_bucket on a synthetic
    nested data dict and returns a sane confusion.
(3) tune_model_params (model-hyperparameter CEM, model-only search on cached
    features) improves validation F1 over a deliberately-bad fixed baseline.
(4) reward modes compute and order as expected.
"""
import os
import sys
from unittest.mock import MagicMock

import torch

# The pipeline imports `ROOT` at module load (dataset.datasets) for feature
# extraction; the tuning code never calls it, so stub it so the import chain
# loads on a machine without MEGAlib.
sys.modules.setdefault("ROOT", MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from pipeline.tuning import (  # noqa: E402
    CEM,
    lean_fit_eval,
    reward,
    tune_model_params,
)

G = torch.Generator().manual_seed(7)


def _half(scale, n):
    return torch.abs(torch.randn(n, generator=G) * scale)


def _feat(cls, bucket, n):
    """Synthetic per-(class,bucket) features. Signal clusters near 0 on delta_E;
    background is mostly spread. Unused features for a bucket are NaN."""
    nan = torch.full((n,), float("nan"))
    if cls == "bdecay":
        dE = _half(0.6, n)
        arm = _half(0.05, n)
        anni = torch.randn(n, generator=G) * 0.2 - 0.9
    else:
        # 30% coincidental (overlap near 0) + 70% spread
        k = int(0.3 * n)
        dE = torch.cat([_half(0.6, k), torch.rand(n - k, generator=G) * 40.0])
        arm = _half(0.3, n)
        anni = torch.randn(n, generator=G) * 0.4 - 0.3
    if bucket == 1:
        arm, anni = nan, nan
    elif bucket == 2:
        anni = nan
    return {"delta_E": dE, "arm": arm, "anni": anni}


def make_data(scale):
    sizes = {("bdecay", 1): 5, ("bg", 1): 3000, ("bdecay", 2): 50, ("bg", 2): 300,
             ("bdecay", 3): 550, ("bg", 3): 1300}
    return {cls: {b: _feat(cls, b, int(scale * sizes[(cls, b)])) for b in (1, 2, 3)}
            for cls in ("bdecay", "bg")}


def test_cem_quadratic():
    space = [("x", -5.0, 5.0, "float"), ("y", -5.0, 5.0, "float")]
    target = [2.3, -1.4]
    cem = CEM(space, pop=30, generations=30, seed=1)
    res = cem.optimize(lambda v: -((v[0] - target[0]) ** 2 + (v[1] - target[1]) ** 2))
    err = ((res["best_vec"][0] - target[0]) ** 2 + (res["best_vec"][1] - target[1]) ** 2) ** 0.5
    print(f"(1) CEM quadratic: best={[round(x,3) for x in res['best_vec']]} target={target} err={err:.4f}")
    assert err < 0.05, "CEM did not converge"
    print("    PASS")


def test_lean_fit_eval():
    fit, val = make_data(4.0), make_data(1.0)
    fit_by, val_by = lean_fit_eval(fit, val, {1: 25, 2: 8, 3: 35}, 0.5, 0.0)
    f1 = reward(fit_by, val_by, "f1")
    print(f"(2) lean_fit_eval: val by-bucket={ {b: (c['tp'],c['fp'],c['fn'],c['tn']) for b,c in val_by.items()} }")
    print(f"    val F1 (bins 25/8/35) = {f1:.4f}")
    assert 0.0 < f1 <= 1.0 and len(val_by) == 3
    print("    PASS")


def test_tune_improves():
    fit, val = make_data(4.0), make_data(1.0)
    # deliberately bad baseline: very coarse bins
    bad_by = lean_fit_eval(fit, val, {1: 8, 2: 4, 3: 10}, 0.0, 0.0)
    bad_f1 = reward(*bad_by, "f1")
    res = tune_model_params(fit, val, mode="f1", pop=16, generations=12, seed=3)
    print(f"(3) tune_model_params: baseline(coarse) F1={bad_f1:.4f} -> tuned F1={res['best_reward']:.4f}")
    print(f"    best params = {res['best_params']}")
    assert res["best_reward"] >= bad_f1, "tuning did not improve over the bad baseline"
    print("    PASS")


def test_reward_modes():
    fit, val = make_data(4.0), make_data(1.0)
    by = lean_fit_eval(fit, val, {1: 25, 2: 8, 3: 35}, 0.5, 0.0)
    f1 = reward(*by, "f1")
    mb = reward(*by, "mult_bias")
    comb = reward(*by, "f1_minus_bias", lam=0.5)
    print(f"(4) reward modes: f1={f1:.4f}  mult_bias(reward)={mb:.4f}  f1_minus_bias={comb:.4f}")
    assert mb <= 0.0  # it's a negative magnitude
    assert abs(comb - (f1 + 0.5 * mb)) < 1e-9
    print("    PASS")


if __name__ == "__main__":
    test_cem_quadratic()
    test_lean_fit_eval()
    test_tune_improves()
    test_reward_modes()
    print("\nALL TUNING CHECKS PASSED")
