import dataclasses
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
)
from pipeline.eval import Evaluator

# ---------------------------------------------------------------------------
# Synthetic stand-in — just enough attributes for Evaluator to work
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SyntheticTrainerLike:
    """Mirrors the attributes Evaluator reads off Trainer."""

    deltaE_angle_bins: torch.Tensor
    angle_bins: torch.Tensor
    deltaE_arm_bins: torch.Tensor
    arm_bins: torch.Tensor
    bdecay_marginal_deltaE: torch.Tensor
    bg_marginal_deltaE: torch.Tensor
    bdecay_cond_angle: torch.Tensor
    bdecay_cond_arm: torch.Tensor
    bg_cond_angle: torch.Tensor
    bg_cond_arm: torch.Tensor
    n_beta_decay: int
    n_bg: int


def _build_synthetic_trainerlike(seed: int = 0):
    """Fit a tiny matrix set on toy data so we can drive Evaluator without
    depending on MEGAlib. Mirrors Trainer.fit's production recipe at 20×20."""
    g = torch.Generator().manual_seed(seed)
    n_b, n_g = 400, 400
    bdecay_dE = torch.rand(n_b, generator=g) * 0.3 + 0.01
    bdecay_angle = -1.0 + torch.rand(n_b, generator=g) * 0.3
    bdecay_arm = torch.rand(n_b, generator=g) * 0.02 + 0.001
    bg_dE = torch.rand(n_g, generator=g) * 4.0 + 1.0
    bg_angle = torch.rand(n_g, generator=g) * 2.0 - 1.0
    bg_arm = torch.rand(n_g, generator=g) * 1.0 + 0.05

    combined_dE = torch.cat([bdecay_dE, bg_dE])
    combined_angle = torch.cat([bdecay_angle, bg_angle])
    combined_arm = torch.cat([bdecay_arm, bg_arm])

    _, dEa_bins, a_bins = build_density_matrix(
        combined_dE,
        combined_angle,
        n_bins_x=20,
        n_bins_y=20,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=1e-3,
    )
    _, dEarm_bins, arm_bins = build_density_matrix(
        combined_dE,
        combined_arm,
        n_bins_x=20,
        n_bins_y=20,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=1e-3,
        log_y_floor=1e-3,
    )
    bdecay_j_a, _, _ = build_density_matrix(bdecay_dE, bdecay_angle, x_bins=dEa_bins, y_bins=a_bins)
    bdecay_j_arm, _, _ = build_density_matrix(bdecay_dE, bdecay_arm, x_bins=dEarm_bins, y_bins=arm_bins)
    bg_j_a, _, _ = build_density_matrix(bg_dE, bg_angle, x_bins=dEa_bins, y_bins=a_bins)
    bg_j_arm, _, _ = build_density_matrix(bg_dE, bg_arm, x_bins=dEarm_bins, y_bins=arm_bins)

    bdecay_m, _ = build_density_matrix_1d(bdecay_dE, x_bins=dEa_bins, smoothing=1.0)
    bg_m, _ = build_density_matrix_1d(bg_dE, x_bins=dEa_bins, smoothing=1.0)

    trainer = _SyntheticTrainerLike(
        deltaE_angle_bins=dEa_bins,
        angle_bins=a_bins,
        deltaE_arm_bins=dEarm_bins,
        arm_bins=arm_bins,
        bdecay_marginal_deltaE=bdecay_m,
        bg_marginal_deltaE=bg_m,
        bdecay_cond_angle=conditional_from_joint(bdecay_j_a, axis=0),
        bdecay_cond_arm=conditional_from_joint(bdecay_j_arm, axis=0),
        bg_cond_angle=conditional_from_joint(bg_j_a, axis=0),
        bg_cond_arm=conditional_from_joint(bg_j_arm, axis=0),
        n_beta_decay=n_b,
        n_bg=n_g,
    )

    data_tuple = (
        torch.cat([torch.ones(n_b, dtype=torch.bool), torch.zeros(n_g, dtype=torch.bool)]),
        bdecay_dE,
        bdecay_angle,
        bdecay_arm,
        bg_dE,
        bg_angle,
        bg_arm,
        combined_dE,
        combined_angle,
        combined_arm,
    )
    return trainer, data_tuple


def _evaluate_and_capture(trainer, data_tuple, split_name: str) -> dict:
    """Run Evaluator.evaluate and return the dict forwarded to wandb.log."""
    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        Evaluator(trainer).evaluate(data_tuple, split_name=split_name)
    finally:
        wandb.log = real_log
    return captured


# ---------------------------------------------------------------------------
# Synthetic-data unit tests
# ---------------------------------------------------------------------------


def test_evaluate_runs_without_crash(wandb_run):
    trainer, data = _build_synthetic_trainerlike()
    Evaluator(trainer).evaluate(data, split_name="synth_eval")


def test_evaluate_logs_metrics_under_requested_split_name(wandb_run):
    """The split_name is the W&B key prefix. Changing it must redirect the
    logged keys correspondingly."""
    trainer, data = _build_synthetic_trainerlike()
    captured = _evaluate_and_capture(trainer, data, split_name="holdout_42")

    for key in [
        "holdout_42/TP",
        "holdout_42/FP",
        "holdout_42/FN",
        "holdout_42/TN",
        "holdout_42/precision",
        "holdout_42/recall",
        "holdout_42/false_positive_rate",
        "holdout_42/f1_score",
        "holdout_42/confusion_matrix",
    ]:
        assert key in captured, f"missing logged key {key!r}"


def test_evaluate_self_on_train_yields_high_f1(wandb_run):
    """Re-evaluating the synthetic trainer on its own training data must
    yield high F1 (≥ 0.9) — the in-sample memorization ceiling."""
    trainer, data = _build_synthetic_trainerlike()
    captured = _evaluate_and_capture(trainer, data, split_name="synth_self")
    assert captured["synth_self/f1_score"] > 0.9, (
        f"synthetic self-eval F1 = {captured['synth_self/f1_score']:.3f} — in-sample fit should be near-perfect"
    )


def test_evaluate_metrics_satisfy_confusion_identities(wandb_run):
    """TP + FN must equal the positive-class count; FP + TN must equal the
    negative-class count. Guards against silent event-drops in evaluate()."""
    trainer, data = _build_synthetic_trainerlike()
    captured = _evaluate_and_capture(trainer, data, split_name="synth_count_check")

    n_pos = int(data[0].sum().item())
    n_neg = int((~data[0]).sum().item())
    tp = captured["synth_count_check/TP"]
    fn = captured["synth_count_check/FN"]
    fp = captured["synth_count_check/FP"]
    tn = captured["synth_count_check/TN"]
    assert tp + fn == n_pos, "TP + FN must equal the positive class count"
    assert fp + tn == n_neg, "FP + TN must equal the negative class count"


def test_evaluator_stores_trainer_reference(wandb_run):
    trainer, _ = _build_synthetic_trainerlike()
    ev = Evaluator(trainer)
    assert ev.trainer is trainer


# ---------------------------------------------------------------------------
# Real-data parametrized tests over BIN_SIZES
# ---------------------------------------------------------------------------


def test_real_evaluate_eval_split_runs(real_trainer, wandb_run):
    """Smoke: Evaluator.evaluate on the eval split completes at every bin
    size and emits at least the f1_score metric under the requested prefix."""
    n_bins = real_trainer["n_bins"]
    captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"test_real_eval_n_bins={n_bins}",
    )
    assert f"test_real_eval_n_bins={n_bins}/f1_score" in captured


def test_real_evaluate_eval_f1_beats_prevalence(real_trainer, wandb_run):
    """Eval-F1 must beat the eval-split prevalence baseline at every n_bins.
    If this fails, the classifier is worse than always predicting the
    majority class."""
    n_bins = real_trainer["n_bins"]
    captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"test_real_eval_beats_prev_n_bins={n_bins}",
    )
    f1 = captured[f"test_real_eval_beats_prev_n_bins={n_bins}/f1_score"]
    gt = real_trainer["eval_tuple"][0]
    prevalence = float(gt.float().mean().item())
    wandb_run.log(
        {
            f"evaluator/n_bins={n_bins}/eval_f1": f1,
            f"evaluator/n_bins={n_bins}/prevalence": prevalence,
        }
    )
    assert f1 > prevalence, (
        f"At n_bins={n_bins}: eval F1 ({f1:.3f}) did not beat prevalence baseline ({prevalence:.3f})"
    )


def test_real_train_eval_f1_gap_diagnostic(real_trainer, wandb_run):
    """Overfitting diagnostic: in-sample train F1 should exceed held-out
    eval F1 at every bin resolution (since the trainer's matrices are
    built from the train data). Log both and the gap under a bin-size
    prefix so the gap's widening with n_bins is visible in W&B."""
    n_bins = real_trainer["n_bins"]
    train_captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["train_tuple"],
        split_name=f"gap/n_bins={n_bins}/train",
    )
    eval_captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"gap/n_bins={n_bins}/eval",
    )
    f1_train = train_captured[f"gap/n_bins={n_bins}/train/f1_score"]
    f1_eval = eval_captured[f"gap/n_bins={n_bins}/eval/f1_score"]
    gap = f1_train - f1_eval

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["train", "eval"], [f1_train, f1_eval], color=["tab:blue", "tab:orange"])
    for bar, v in zip(bars, [f1_train, f1_eval]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(f"Train vs eval F1 (n_bins={n_bins}, gap={gap:.3f})")

    wandb_run.log(
        {
            f"evaluator/gap/n_bins={n_bins}/train_f1": f1_train,
            f"evaluator/gap/n_bins={n_bins}/eval_f1": f1_eval,
            f"evaluator/gap/n_bins={n_bins}/f1_gap_train_minus_eval": gap,
            f"evaluator/gap/n_bins={n_bins}/bars": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert f1_train >= f1_eval - 0.02, (
        f"At n_bins={n_bins}: train F1 ({f1_train:.3f}) is below eval F1 ({f1_eval:.3f}) — "
        "the trainer's matrices were built from train, so in-sample fit should be at least as good"
    )


def test_real_eval_predicted_label_distribution_visualisation(real_trainer, wandb_run):
    """Plot a side-by-side comparison of predicted vs actual label counts
    on the eval split. Reveals class bias at a glance."""
    n_bins = real_trainer["n_bins"]
    eval_tuple = real_trainer["eval_tuple"]
    captured = _evaluate_and_capture(
        real_trainer["trainer"],
        eval_tuple,
        split_name=f"label_dist/n_bins={n_bins}",
    )

    tp = captured[f"label_dist/n_bins={n_bins}/TP"]
    fp = captured[f"label_dist/n_bins={n_bins}/FP"]
    fn = captured[f"label_dist/n_bins={n_bins}/FN"]
    tn = captured[f"label_dist/n_bins={n_bins}/TN"]
    n_pred_beta = tp + fp
    n_pred_bg = fn + tn
    n_true_beta = tp + fn
    n_true_bg = fp + tn

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [n_true_beta, n_true_bg], width, label="true")
    ax.bar(x + width / 2, [n_pred_beta, n_pred_bg], width, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(["β", "bg"])
    ax.set_ylabel("count (eval split)")
    ax.set_title(f"Eval-split predicted vs true class counts (n_bins={n_bins})")
    ax.legend()

    wandb_run.log({f"evaluator/n_bins={n_bins}/label_distribution": wandb.Image(fig)})
    plt.close(fig)


def test_real_eval_metric_sweep_visualisation(real_trainer, wandb_run):
    """Plot precision / recall / F1 / FPR at this n_bins. Useful for
    spotting recall-dominant or precision-dominant failure modes."""
    n_bins = real_trainer["n_bins"]
    captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"metric_sweep/n_bins={n_bins}",
    )
    precision = captured[f"metric_sweep/n_bins={n_bins}/precision"]
    recall = captured[f"metric_sweep/n_bins={n_bins}/recall"]
    f1 = captured[f"metric_sweep/n_bins={n_bins}/f1_score"]
    fpr = captured[f"metric_sweep/n_bins={n_bins}/false_positive_rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["precision", "recall", "f1", "fpr"], [precision, recall, f1, fpr])
    for bar, v in zip(bars, [precision, recall, f1, fpr]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("metric")
    ax.set_title(f"Evaluator metric sweep (n_bins={n_bins})")
    wandb_run.log({f"evaluator/n_bins={n_bins}/metric_bars": wandb.Image(fig)})
    plt.close(fig)


def test_real_eval_confusion_identities_hold(real_trainer, wandb_run):
    """TP+FP+FN+TN must exactly equal the eval-split size (no events dropped
    on the floor). Precision and recall must lie in [0, 1]."""
    n_bins = real_trainer["n_bins"]
    captured = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"confusion_sanity/n_bins={n_bins}",
    )
    tp = captured[f"confusion_sanity/n_bins={n_bins}/TP"]
    fp = captured[f"confusion_sanity/n_bins={n_bins}/FP"]
    fn = captured[f"confusion_sanity/n_bins={n_bins}/FN"]
    tn = captured[f"confusion_sanity/n_bins={n_bins}/TN"]
    total = tp + fp + fn + tn
    eval_size = real_trainer["eval_tuple"][0].numel()
    assert total == eval_size, f"confusion-cell total ({total}) != eval size ({eval_size})"

    precision = captured[f"confusion_sanity/n_bins={n_bins}/precision"]
    recall = captured[f"confusion_sanity/n_bins={n_bins}/recall"]
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


def test_real_evaluate_repeat_calls_deterministic(real_trainer, wandb_run):
    """Calling Evaluator.evaluate() twice on the same data + trainer must
    produce identical metrics — no stochasticity in inference."""
    n_bins = real_trainer["n_bins"]
    first = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"repeat/n_bins={n_bins}/a",
    )
    second = _evaluate_and_capture(
        real_trainer["trainer"],
        real_trainer["eval_tuple"],
        split_name=f"repeat/n_bins={n_bins}/b",
    )
    for metric in ["f1_score", "precision", "recall", "false_positive_rate"]:
        a = first[f"repeat/n_bins={n_bins}/a/{metric}"]
        b = second[f"repeat/n_bins={n_bins}/b/{metric}"]
        assert math.isclose(a, b, abs_tol=1e-12), f"{metric} changed between repeat evaluate() calls"


def test_real_eval_data_tuple_shape_contract(real_trainer):
    """Evaluator.evaluate unpacks a 10-tuple — if the tuple shape from
    Datasets.split_dataset drifts, evaluate() breaks."""
    eval_tuple = real_trainer["eval_tuple"]
    assert len(eval_tuple) == 10
    assert isinstance(eval_tuple[0], torch.Tensor)
    assert eval_tuple[0].dtype == torch.bool
