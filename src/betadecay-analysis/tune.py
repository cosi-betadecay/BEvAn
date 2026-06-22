"""In-pipeline hyperparameter optimization entry point (the "RL loop in the code").

Mirrors run.py's setup, then:
  1. computes features once (default feature params),
  2. carves a fit/val split out of TRAIN (eval stays sealed),
  3. runs the CEM optimizer over the MODEL hyperparameters, scoring on val,
  4. reports an A/B on the SEALED eval set: champion-default params vs tuned params,
     each run exactly once.

Run (on the MEGAlib VM):
    python3 src/betadecay-analysis/tune.py
Env knobs:
    BETADECAY_TUNE_REWARD       f1 | mult_bias | f1_minus_bias   (default f1)
    BETADECAY_TUNE_GENERATIONS  CEM generations                  (default 20)
    BETADECAY_TUNE_POP          CEM population per generation     (default 24)
    BETADECAY_TUNE_LAMBDA       bias weight for f1_minus_bias     (default 0.5)

Feature-level params (B2B_*/ARM) are supported by pipeline.tuning
(FEATURE_PARAM_SPACE / apply_feature_params) but require a feature recompute per
trial; this entry point tunes the cheap model params on cached features first.
"""
import os

import hydra
import omegaconf
import pipeline.train as train_mod
from dataset.datasets import Datasets
from dotenv import load_dotenv
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from pipeline.tuning import tune_model_params
from utils.reader_extraction import get_reader

import wandb

DEFAULT_PARAMS = {"n_bins": {1: 25, 2: 8, 3: 35}, "joint_smooth": 0.5, "marg_smooth": 0.0}


def _apply(params):
    train_mod._N_BINS = dict(params["n_bins"])
    train_mod._JOINT_SMOOTHING = float(params["joint_smooth"])
    train_mod._MARGINAL_SMOOTHING = float(params["marg_smooth"])


def _eval_f1(cfg, train, evaluate, params, tag):
    """Refit on full train with `params`, evaluate ONCE on the sealed eval set."""
    _apply(params)
    trainer = Trainer(cfg).fit(train)
    res = Evaluator(trainer).evaluate(evaluate, split_name=f"eval_{tag}")
    t = res["total"]
    prec = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
    rec = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"  [{tag}] eval F1={f1:.4f}  params={params}")
    return f1


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    wandb.init(project=cfg.project_names[0])
    reward_mode = os.getenv("BETADECAY_TUNE_REWARD", "f1")
    generations = int(os.getenv("BETADECAY_TUNE_GENERATIONS", "20"))
    pop = int(os.getenv("BETADECAY_TUNE_POP", "24"))
    lam = float(os.getenv("BETADECAY_TUNE_LAMBDA", "0.5"))
    ref_energy = 511.0

    imager = FarFieldImager(
        geometry_file=cfg.setup.geo_file, n_phi=360, n_theta=180, coordinate_system="spheric"
    )
    imager.backproject_file(cfg.setup.tra_file)
    unit_vector = imager.peak_direction_cartesian()
    reader_sim, reader_tra = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    datasets = Datasets(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    data = datasets.compute_event_features(cfg, ref_energy, reader_sim, reader_tra, unit_vector)

    # eval is SEALED; carve fit/val out of TRAIN for the optimization reward.
    train, evaluate = datasets.split_dataset(data)
    fit, val = datasets.split_dataset(train)

    print(f"\nCEM tuning: reward={reward_mode} pop={pop} generations={generations}")

    def _log(h):
        print(f"  gen {h['gen']:02d}: best={h['best_r']:.4f}  gen_best={h['gen_best']:.4f}  "
              f"gen_mean={h['gen_mean']:.4f}")
        wandb.log({"tune/best_reward": h["best_r"], "tune/gen_best": h["gen_best"],
                   "tune/gen_mean": h["gen_mean"], "tune/generation": h["gen"]})

    res = tune_model_params(fit, val, mode=reward_mode, lam=lam, pop=pop,
                            generations=generations, log=_log)
    tuned = res["best_params"]
    print(f"\nBest val reward ({reward_mode}) = {res['best_reward']:.4f}  params={tuned}")

    # A/B on the SEALED eval set: champion default vs tuned, each run once.
    print("\nFinal A/B on sealed eval set:")
    f1_default = _eval_f1(cfg, train, evaluate, DEFAULT_PARAMS, "default")
    f1_tuned = _eval_f1(cfg, train, evaluate, tuned, "tuned")
    print(f"\n  champion-default eval F1 = {f1_default:.4f}")
    print(f"  tuned            eval F1 = {f1_tuned:.4f}   (delta {f1_tuned - f1_default:+.4f})")
    if f1_tuned <= f1_default + 1e-4:
        print("  -> tuning did NOT beat the default: consistent with the per-event ceiling.")
    wandb.log({"tune/eval_f1_default": f1_default, "tune/eval_f1_tuned": f1_tuned,
               "tune/eval_f1_delta": f1_tuned - f1_default})
    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")
    wandb.login(key=wandb_api_key)
    main()
