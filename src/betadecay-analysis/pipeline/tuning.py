"""In-pipeline black-box optimization ("RL step") over the classifier's
hyperparameters, scored on a validation split carved from TRAIN. Eval is never
touched here -- the caller runs it once at the end with the returned best params.

The pipeline is non-differentiable (histogram densities, argmin-over-subsets,
hard bucketing), so there is no gradient: the optimizer is the Cross-Entropy
Method (CEM), a standard black-box / policy-search update. Each "episode"
proposes a parameter vector, fits the per-bucket density model on the fit subset,
evaluates it on the validation subset, and returns a scalar reward.

Two parameter classes:
  * MODEL params (_N_BINS per bucket, joint/marginal smoothing) -- change only the
    density model, NOT the features, so the SAME cached feature tensors are reused
    every episode (cheap; hundreds of episodes per minute).
  * FEATURE params (B2B_*/ARM constants in physics_factors) -- change the features,
    so they require a feature recompute; the caller (tune.py) handles that as a
    slower outer loop and passes freshly-computed fit/val dicts in.

To guarantee the loop can never drift from production behavior, the fit/eval
reuses the REAL ``Trainer._fit_bucket`` and ``Evaluator._evaluate_bucket`` by
temporarily setting the three hyperparameter globals in ``pipeline.train``.

Reward modes (higher is better):
  * ``f1``           : pooled validation F1.
  * ``mult_bias``    : negative multiplicity-DIFFERENTIAL count bias --
                       sum_b |S_hat_b - true_b| / sum_b true_b over constrained
                       buckets. Harder to game than the pooled count bias (whose
                       FP/FN cancel), per the response-matrix discussion.
  * ``f1_minus_bias``: f1 - lambda * (mult_bias magnitude).
"""

from dataset.datasets import BUCKETS
from modeling.population_correction import population_estimate

import pipeline.train as train_mod
from pipeline.eval import Evaluator

# --- reward ----------------------------------------------------------------

def _f1(by_bucket: dict) -> float:
    tp = sum(c["tp"] for c in by_bucket.values())
    fp = sum(c["fp"] for c in by_bucket.values())
    fn = sum(c["fn"] for c in by_bucket.values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _mult_bias_magnitude(fit_by_bucket: dict, val_by_bucket: dict) -> float:
    """Multiplicity-differential count bias: sum_b |S_hat_b - true_b| / sum true_b
    over buckets the correction can constrain (rates from fit, applied to val)."""
    est = population_estimate(fit_by_bucket, val_by_bucket)
    num = 0.0
    den = 0.0
    for r in est["rows"]:
        if r["constrained"]:
            num += abs(r["s_hat"] - r["true_s"])
            den += r["true_s"]
    return (num / den) if den else float("inf")


def reward(fit_by_bucket: dict, val_by_bucket: dict, mode: str = "f1", lam: float = 0.5) -> float:
    if mode == "f1":
        return _f1(val_by_bucket)
    if mode == "mult_bias":
        return -_mult_bias_magnitude(fit_by_bucket, val_by_bucket)
    if mode == "f1_minus_bias":
        return _f1(val_by_bucket) - lam * _mult_bias_magnitude(fit_by_bucket, val_by_bucket)
    raise ValueError(f"unknown reward mode {mode!r}")


# --- environment: fit on fit-subset, evaluate on fit and val ---------------

def lean_fit_eval(fit_data: dict, val_data: dict, n_bins: dict, joint_smooth: float, marg_smooth: float):
    """Fit per-bucket models on ``fit_data`` with the given MODEL hyperparameters
    and return ``(fit_by_bucket, val_by_bucket)`` confusion dicts. Reuses the real
    production fit/eval code by patching the three ``pipeline.train`` globals; the
    patch is always restored. No wandb / no logging side effects."""
    saved = (train_mod._N_BINS, train_mod._JOINT_SMOOTHING, train_mod._MARGINAL_SMOOTHING)
    try:
        train_mod._N_BINS = dict(n_bins)
        train_mod._JOINT_SMOOTHING = float(joint_smooth)
        train_mod._MARGINAL_SMOOTHING = float(marg_smooth)
        trainer = train_mod.Trainer(cfg=None)
        trainer.models = {
            b: trainer._fit_bucket(fit_data["bdecay"][b], fit_data["bg"][b], b) for b in BUCKETS
        }
        ev = Evaluator(trainer)
        fit_by = {}
        val_by = {}
        for b in BUCKETS:
            cf = ev._evaluate_bucket(fit_data, b, trainer.models[b])
            cv = ev._evaluate_bucket(val_data, b, trainer.models[b])
            if cf is not None:
                fit_by[b] = cf
            if cv is not None:
                val_by[b] = cv
        return fit_by, val_by
    finally:
        train_mod._N_BINS, train_mod._JOINT_SMOOTHING, train_mod._MARGINAL_SMOOTHING = saved


# --- parameter space (MODEL params) ----------------------------------------
# Each entry: (name, low, high, kind). 'int' params are rounded at decode time.
MODEL_PARAM_SPACE = [
    ("n_bins_1", 8.0, 40.0, "int"),
    ("n_bins_2", 4.0, 30.0, "int"),
    ("n_bins_3", 10.0, 60.0, "int"),
    ("joint_smoothing", 0.0, 2.0, "float"),
    ("marginal_smoothing", 0.0, 1.0, "float"),
]


def decode_model_params(vec) -> dict:
    """Map a CEM vector (in MODEL_PARAM_SPACE order) to fit-eval kwargs."""
    d = {}
    for (name, lo, hi, kind), v in zip(MODEL_PARAM_SPACE, vec, strict=False):
        v = min(max(float(v), lo), hi)
        d[name] = int(round(v)) if kind == "int" else v
    return {
        "n_bins": {1: d["n_bins_1"], 2: d["n_bins_2"], 3: d["n_bins_3"]},
        "joint_smooth": d["joint_smoothing"],
        "marg_smooth": d["marginal_smoothing"],
    }


# --- CEM optimizer ----------------------------------------------------------

class CEM:
    """Cross-Entropy Method over a box-bounded continuous space.

    Maintains a diagonal Gaussian over the parameter vector; each generation
    samples ``pop`` candidates, keeps the top ``elite_frac`` by reward, and
    refits the Gaussian to the elites. Deterministic given ``seed`` (no global
    RNG; uses a torch.Generator) so runs are reproducible.
    """

    def __init__(self, space, pop=24, elite_frac=0.25, generations=25, seed=0, init_std_frac=0.5):
        import torch
        self.space = space
        self.lo = torch.tensor([s[1] for s in space], dtype=torch.float64)
        self.hi = torch.tensor([s[2] for s in space], dtype=torch.float64)
        self.pop = pop
        self.n_elite = max(2, int(round(pop * elite_frac)))
        self.generations = generations
        self.gen = torch.Generator().manual_seed(seed)
        self.mean = (self.lo + self.hi) / 2.0
        self.std = (self.hi - self.lo) * init_std_frac
        self._torch = torch

    def optimize(self, reward_fn, log=None):
        torch = self._torch
        best_vec, best_r = None, -float("inf")
        history = []
        for g in range(self.generations):
            z = torch.randn(self.pop, len(self.space), generator=self.gen, dtype=torch.float64)
            samples = self.mean + z * self.std
            samples = torch.maximum(torch.minimum(samples, self.hi), self.lo)
            rewards = []
            for i in range(self.pop):
                r = reward_fn(samples[i].tolist())
                rewards.append(r)
                if r > best_r:
                    best_r, best_vec = r, samples[i].tolist()
            rewards_t = torch.tensor(rewards, dtype=torch.float64)
            elite_idx = torch.topk(rewards_t, self.n_elite).indices
            elite = samples[elite_idx]
            self.mean = elite.mean(dim=0)
            self.std = elite.std(dim=0).clamp_min(1e-6)
            history.append({"gen": g, "best_r": best_r, "gen_best": float(rewards_t.max()),
                            "gen_mean": float(rewards_t.mean())})
            if log:
                log(history[-1])
        return {"best_vec": best_vec, "best_reward": best_r, "history": history}


def tune_model_params(fit_data, val_data, mode="f1", lam=0.5, pop=24, generations=25, seed=0, log=None):
    """Run CEM over the MODEL hyperparameters on cached features. Returns the best
    decoded params and reward. Features are NOT recomputed (model-only search)."""
    def reward_fn(vec):
        p = decode_model_params(vec)
        fit_by, val_by = lean_fit_eval(fit_data, val_data, p["n_bins"], p["joint_smooth"], p["marg_smooth"])
        return reward(fit_by, val_by, mode=mode, lam=lam)

    cem = CEM(MODEL_PARAM_SPACE, pop=pop, generations=generations, seed=seed)
    result = cem.optimize(reward_fn, log=log)
    result["best_params"] = decode_model_params(result["best_vec"])
    return result


# --- FEATURE params (applied by the caller before a feature recompute) ------
# physics_factors module constants; monkey-patched by tune.py, never hand-edited.
FEATURE_PARAM_SPACE = [
    ("B2B_E_FLOOR", 480.0, 511.0, "float"),
    ("B2B_SIGMA_E", 100.0, 400.0, "float"),
    ("B2B_LAMBDA", 0.0, 1.0, "float"),
]


def decode_feature_params(vec) -> dict:
    d = {}
    for (name, lo, hi, _), v in zip(FEATURE_PARAM_SPACE, vec, strict=False):
        d[name] = min(max(float(v), lo), hi)
    return d


def apply_feature_params(params: dict):
    """Patch physics_factors module constants in place (returns the saved values
    so the caller can restore them)."""
    import physics.physics_factors as pf
    saved = {k: getattr(pf, k) for k in params}
    for k, v in params.items():
        setattr(pf, k, v)
    return saved
