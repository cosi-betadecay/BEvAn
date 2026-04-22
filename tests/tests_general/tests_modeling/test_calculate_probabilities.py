"""
Unit + real-data tests for modeling/calculate_probablities.py.

The module wires together density lookup, the likelihood-ratio R, and the
predict() endpoint that turns ratios into W&B-logged classification metrics.

Coverage:
    - R(...)             — pure log-ratio composition; synthetic + real
    - get_probabilities  — six lookups against pre-built matrices
    - predict(...)       — smoke + real-data F1 attribution sweeps

Real-data sweeps are parametrized over BIN_SIZES = [20, 200, 1000] via the
``likelihood_tensors`` fixture in tests_general/conftest.py.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from modeling.calculate_probablities import R, get_probabilities, predict


# ---------------------------------------------------------------------------
# R() — pure log-ratio composition
# ---------------------------------------------------------------------------


def test_R_balanced_likelihoods_returns_one():
    """All β/bg pairs equal ⇒ log-ratio = 0 ⇒ R = 1."""
    p = torch.tensor([0.1, 0.2, 0.3])
    out = R(p, p, p, p, p, p)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_R_beta_factors_dominant_returns_above_one():
    p_beta = torch.tensor([0.5, 0.5, 0.5])
    p_bg = torch.tensor([0.1, 0.1, 0.1])
    out = R(p_beta, p_bg, p_beta, p_bg, p_beta, p_bg)
    assert torch.all(out > 1.0)


def test_R_bg_factors_dominant_returns_below_one():
    p_beta = torch.tensor([0.1, 0.1, 0.1])
    p_bg = torch.tensor([0.5, 0.5, 0.5])
    out = R(p_beta, p_bg, p_beta, p_bg, p_beta, p_bg)
    assert torch.all(out < 1.0)


def test_R_uses_eps_to_avoid_log_zero():
    """A zero likelihood (NaN/OOD lookup) must produce a finite R via the
    `eps` floor in each log call rather than -inf."""
    zeros = torch.zeros(3)
    ones = torch.ones(3)
    out = R(zeros, ones, zeros, ones, zeros, ones)
    assert torch.all(torch.isfinite(out))


def test_R_factorises_as_product_of_subratios():
    """log R = (log P_β/P_bg)_ΔE + (log)_angle + (log)_arm  ⇒
       R = R_ΔE · R_angle · R_arm."""
    p1 = torch.tensor([0.4])
    q1 = torch.tensor([0.1])
    p2 = torch.tensor([0.3])
    q2 = torch.tensor([0.2])
    p3 = torch.tensor([0.5])
    q3 = torch.tensor([0.25])
    full = R(p1, q1, p2, q2, p3, q3)
    pieces = ((p1 + 1e-8) / (q1 + 1e-8)) * ((p2 + 1e-8) / (q2 + 1e-8)) * ((p3 + 1e-8) / (q3 + 1e-8))
    assert torch.allclose(full, pieces, rtol=1e-5)


def test_R_returns_strictly_positive():
    """R is exp(log_r) — must always be > 0."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand(50, generator=g) * 0.5
    b = torch.rand(50, generator=g) * 0.5
    out = R(a, b, a, b, a, b)
    assert torch.all(out > 0.0)


def test_R_supports_higher_dim_inputs():
    """The op is element-wise; multi-dim tensors must broadcast cleanly."""
    a = torch.rand(4, 5)
    b = torch.rand(4, 5)
    out = R(a, b, a, b, a, b)
    assert out.shape == (4, 5)
    assert torch.all(torch.isfinite(out))


def test_R_eps_parameter_changes_zero_floor():
    """Larger eps shifts the zero-input ratio toward 1 (the limit eps→large)."""
    zeros = torch.zeros(1)
    ones = torch.ones(1)
    small_eps = R(zeros, ones, zeros, ones, zeros, ones, eps=1e-8)
    big_eps = R(zeros, ones, zeros, ones, zeros, ones, eps=1.0)
    # eps=1e-8: ratio ≈ ((1e-8)/1)^3 ≈ 1e-24 (effectively 0).
    # eps=1.0:  ratio ≈ (1/2)^3 = 0.125 — much larger.
    assert big_eps.item() > small_eps.item()


# ---------------------------------------------------------------------------
# get_probabilities() — wraps the six lookups
# ---------------------------------------------------------------------------


def test_get_probabilities_returns_six_tensors_of_correct_length():
    """Build a tiny set of density matrices manually and confirm
    get_probabilities returns six per-event tensors all shaped like the
    gen inputs."""
    n_x = n_y = 8
    n_gen = 12

    bdecay_marginal_dE = torch.full((n_x,), 1.0 / n_x)
    bg_marginal_dE = torch.full((n_x,), 1.0 / n_x)
    bdecay_cond_angle = torch.full((n_x, n_y), 1.0 / n_y)
    bdecay_cond_arm = torch.full((n_x, n_y), 1.0 / n_y)
    bg_cond_angle = torch.full((n_x, n_y), 1.0 / n_y)
    bg_cond_arm = torch.full((n_x, n_y), 1.0 / n_y)

    deltaE_angle_bins = torch.linspace(0.1, 10.0, n_x + 1)
    angle_bins = torch.linspace(-1.0, 1.0, n_y + 1)
    deltaE_arm_bins = torch.linspace(0.1, 10.0, n_x + 1)
    arm_bins = torch.linspace(0.001, 1.0, n_y + 1)

    g = torch.Generator().manual_seed(7)
    gen_dE = torch.rand(n_gen, generator=g) * 9.0 + 0.5
    gen_angle = torch.rand(n_gen, generator=g) * 2.0 - 1.0
    gen_arm = torch.rand(n_gen, generator=g) * 0.9 + 0.05

    out = get_probabilities(
        bdecay_marginal_dE,
        bg_marginal_dE,
        bdecay_cond_angle,
        bdecay_cond_arm,
        bg_cond_angle,
        bg_cond_arm,
        gen_dE,
        gen_angle,
        gen_arm,
        deltaE_angle_bins,
        angle_bins,
        deltaE_arm_bins,
        arm_bins,
    )
    assert len(out) == 6
    for t in out:
        assert t.numel() == n_gen


def test_get_probabilities_returns_zero_for_nan_gen_inputs():
    """NaN gen inputs propagate to 0.0 in all six outputs — matches the
    contract of lookup_density_values{,_1d}."""
    n_x = n_y = 6
    bdecay_marginal_dE = torch.full((n_x,), 1.0 / n_x)
    bg_marginal_dE = torch.full((n_x,), 1.0 / n_x)
    cond = torch.full((n_x, n_y), 1.0 / n_y)
    deltaE_angle_bins = torch.linspace(0.1, 10.0, n_x + 1)
    angle_bins = torch.linspace(-1.0, 1.0, n_y + 1)
    deltaE_arm_bins = torch.linspace(0.1, 10.0, n_x + 1)
    arm_bins = torch.linspace(0.001, 1.0, n_y + 1)

    gen_dE = torch.tensor([1.0, float("nan"), 3.0])
    gen_angle = torch.tensor([0.5, float("nan"), -0.5])
    gen_arm = torch.tensor([0.2, float("nan"), 0.4])

    out = get_probabilities(
        bdecay_marginal_dE,
        bg_marginal_dE,
        cond,
        cond,
        cond,
        cond,
        gen_dE,
        gen_angle,
        gen_arm,
        deltaE_angle_bins,
        angle_bins,
        deltaE_arm_bins,
        arm_bins,
    )
    for t in out:
        assert t[1].item() == 0.0


# ---------------------------------------------------------------------------
# predict() — wandb side-effect smoke + real-data F1 sweeps
# ---------------------------------------------------------------------------


def test_predict_logs_required_keys_for_synthetic_inputs(wandb_run):
    """predict() should emit TP/FP/FN/TN, precision, recall, fpr, f1_score,
    and a confusion-matrix image — all under the supplied split_name prefix."""
    n_per = 4
    p_beta = torch.tensor([0.9] * n_per + [0.1] * n_per)
    p_bg = torch.tensor([0.1] * n_per + [0.9] * n_per)
    gt = torch.tensor([True] * n_per + [False] * n_per)

    # No exception ⇒ wandb.log accepted the payload under the prefix.
    predict(
        gt,
        p_beta,
        p_bg,
        p_beta,
        p_bg,
        p_beta,
        p_bg,
        n_beta_decay=n_per,
        n_bg=n_per,
        split_name="test_predict_synth",
    )


def test_predict_perfect_classifier_yields_unit_f1(wandb_run):
    """If all true-β events have R≫1 and all true-bg have R≪1, F1 must be 1.
    Capture the F1 by intercepting the wandb.log call."""
    n_per = 32
    p_beta = torch.cat([torch.full((n_per,), 0.99), torch.full((n_per,), 0.01)])
    p_bg = torch.cat([torch.full((n_per,), 0.01), torch.full((n_per,), 0.99)])
    gt = torch.tensor([True] * n_per + [False] * n_per)

    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        predict(
            gt,
            p_beta,
            p_bg,
            p_beta,
            p_bg,
            p_beta,
            p_bg,
            n_beta_decay=n_per,
            n_bg=n_per,
            split_name="test_predict_perfect",
        )
    finally:
        wandb.log = real_log
    assert math.isclose(captured["test_predict_perfect/f1_score"], 1.0, abs_tol=1e-9)


def test_predict_random_classifier_yields_modest_f1(wandb_run):
    """When β / bg likelihoods are independent of the truth label, F1 should
    sit near the prevalence baseline (well below 0.9 for balanced classes)."""
    g = torch.Generator().manual_seed(123)
    n_per = 200
    p_beta = torch.rand(2 * n_per, generator=g)
    p_bg = torch.rand(2 * n_per, generator=g)
    gt = torch.tensor([True] * n_per + [False] * n_per)

    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        predict(
            gt,
            p_beta,
            p_bg,
            p_beta,
            p_bg,
            p_beta,
            p_bg,
            n_beta_decay=n_per,
            n_bg=n_per,
            split_name="test_predict_random",
        )
    finally:
        wandb.log = real_log
    f1 = captured["test_predict_random/f1_score"]
    assert f1 < 0.9, f"random classifier F1 = {f1:.3f} — implausibly high"


# ---------------------------------------------------------------------------
# Real-data tests parametrized over BIN_SIZES
# ---------------------------------------------------------------------------


def test_real_R_distribution_separates_classes(likelihood_tensors, wandb_run):
    """The likelihood ratio R should sit higher (in log space) for true-β
    events than for true-bg events at every bin resolution. Plot the per-class
    log-R distribution."""
    ratio = likelihood_tensors["ratio"].cpu().numpy()
    labels = likelihood_tensors["labels"]
    n_bins = likelihood_tensors["n_bins"]

    log_r = np.log(np.clip(ratio, 1e-300, None))
    valid = np.isfinite(log_r)
    log_r = log_r[valid]
    labels_v = labels[valid]

    log_r_beta = log_r[labels_v == 1]
    log_r_bg = log_r[labels_v == 0]
    assert log_r_beta.size > 0 and log_r_bg.size > 0

    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.linspace(min(log_r.min(), -10), max(log_r.max(), 10), 100)
    ax.hist(log_r_bg, bins=edges, alpha=0.5, density=True, label=f"true bg (n={log_r_bg.size}, μ={log_r_bg.mean():.2f})")
    ax.hist(log_r_beta, bins=edges, alpha=0.5, density=True, label=f"true β (n={log_r_beta.size}, μ={log_r_beta.mean():.2f})")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="R = 1")
    ax.set_xlabel("log R")
    ax.set_ylabel("density")
    ax.set_title(f"Per-class log-likelihood ratio (n_bins={n_bins})")
    ax.legend(loc="upper right", fontsize=8)

    wandb_run.log(
        {
            f"R/n_bins={n_bins}/mean_log_R_beta": float(log_r_beta.mean()),
            f"R/n_bins={n_bins}/mean_log_R_bg": float(log_r_bg.mean()),
            f"R/n_bins={n_bins}/separation_log": float(log_r_beta.mean() - log_r_bg.mean()),
            f"R/n_bins={n_bins}/per_class_hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert log_r_beta.mean() > log_r_bg.mean(), (
        f"At n_bins={n_bins}: true-β mean log R ({log_r_beta.mean():.3f}) "
        f"should exceed true-bg mean log R ({log_r_bg.mean():.3f})"
    )


def test_real_predict_f1_recovered_from_wandb(likelihood_tensors, wandb_run):
    """End-to-end: predict() on the real-data ratio should yield a non-trivial
    F1 (better than the prevalence baseline) at every n_bins. Capture
    metrics from wandb.log to assert."""
    n_bins = likelihood_tensors["n_bins"]
    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        predict(
            torch.as_tensor(likelihood_tensors["labels"], dtype=torch.bool),
            likelihood_tensors["p_beta_deltaE"],
            likelihood_tensors["p_bg_deltaE"],
            likelihood_tensors["p_beta_angle"],
            likelihood_tensors["p_bg_angle"],
            likelihood_tensors["p_beta_arm"],
            likelihood_tensors["p_bg_arm"],
            n_beta_decay=likelihood_tensors["n_beta_decay"],
            n_bg=likelihood_tensors["n_bg"],
            split_name=f"calc_probs_n_bins={n_bins}",
        )
    finally:
        wandb.log = real_log

    f1 = captured[f"calc_probs_n_bins={n_bins}/f1_score"]
    prevalence = float((likelihood_tensors["labels"] == 1).mean())
    assert f1 > prevalence, (
        f"n_bins={n_bins}: F1 ({f1:.3f}) did not beat prevalence baseline ({prevalence:.3f})"
    )


def test_real_R_score_distribution_per_class_visualisation(likelihood_tensors, wandb_run):
    """Plot the raw posterior P(β|D) (Bayesian model output) histogrammed
    per true class. A well-separating classifier produces a bimodal joint
    histogram."""
    n_bins = likelihood_tensors["n_bins"]
    posterior = likelihood_tensors["model"].beta_decay_given_data_probability().cpu().numpy()
    labels = likelihood_tensors["labels"]

    valid = np.isfinite(posterior)
    posterior = posterior[valid]
    labels_v = labels[valid]
    p_beta = posterior[labels_v == 1]
    p_bg = posterior[labels_v == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.linspace(0, 1, 41)
    ax.hist(p_bg, bins=edges, alpha=0.5, density=True, label=f"true bg (n={p_bg.size})")
    ax.hist(p_beta, bins=edges, alpha=0.5, density=True, label=f"true β (n={p_beta.size})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Posterior P(β | D)")
    ax.set_ylabel("density")
    ax.set_title(f"Posterior per class (n_bins={n_bins})")
    ax.legend()

    wandb_run.log({f"calc_probs/n_bins={n_bins}/posterior_per_class": wandb.Image(fig)})
    plt.close(fig)


def test_real_R_decomposition_per_factor(likelihood_tensors, wandb_run):
    """Decompose log R into its three sub-ratios (ΔE marginal, angle|ΔE,
    ARM|ΔE) and plot the per-class distributions side-by-side. Visually
    shows which factor pulls predictions the most."""
    n_bins = likelihood_tensors["n_bins"]
    eps = 1e-8
    factors = {
        "deltaE": (likelihood_tensors["p_beta_deltaE"], likelihood_tensors["p_bg_deltaE"]),
        "angle|ΔE": (likelihood_tensors["p_beta_angle"], likelihood_tensors["p_bg_angle"]),
        "ARM|ΔE": (likelihood_tensors["p_beta_arm"], likelihood_tensors["p_bg_arm"]),
    }
    labels = likelihood_tensors["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, (p_beta, p_bg)) in zip(axes, factors.items()):
        log_sub = (torch.log(p_beta + eps) - torch.log(p_bg + eps)).cpu().numpy()
        valid = np.isfinite(log_sub)
        log_sub_v = log_sub[valid]
        labels_v = labels[valid]
        beta_part = log_sub_v[labels_v == 1]
        bg_part = log_sub_v[labels_v == 0]
        edges = np.linspace(log_sub_v.min(), log_sub_v.max(), 80)
        ax.hist(bg_part, bins=edges, alpha=0.5, density=True, label="true bg")
        ax.hist(beta_part, bins=edges, alpha=0.5, density=True, label="true β")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"log P_β/P_bg ({name})")
        ax.set_ylabel("density")
        ax.set_title(f"{name} sub-ratio (Δμ = {beta_part.mean() - bg_part.mean():.2f})")
        ax.legend()

        wandb_run.log(
            {
                f"R_decomp/{name}/n_bins={n_bins}/separation": float(beta_part.mean() - bg_part.mean()),
            }
        )
    wandb_run.log({f"R_decomp/n_bins={n_bins}/factors": wandb.Image(fig)})
    plt.close(fig)
