"""
Tests for pipeline/train.py — the Trainer class.

Trainer.fit() consumes the 10-tuple emitted by Datasets.split_dataset, builds
shared bin edges from the combined-train tensors, then fills per-class joint
matrices, marginals, and conditionals. It also calls Evaluator(self).evaluate
on the training split as a side-effect, which logs metrics under split_name
"train" to W&B.

The production code hardcodes the matrix bin count to 200×200. To sweep
``BIN_SIZES = [20, 200, 1000]`` we monkeypatch ``pipeline.train.
build_density_matrix`` so each parametrized run uses a different resolution.

Test split:
    - Synthetic-data unit tests (one bin size, focused on contracts)
    - Real-data parametrized tests (full BIN_SIZES sweep, plot fitted
      matrices to W&B for visual inspection)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from pipeline.train import Trainer

BIN_SIZES = [20, 200, 1000]


# ---------------------------------------------------------------------------
# Helpers (the parametrized real_trainer fixture lives in tests_general/conftest.py)
# ---------------------------------------------------------------------------


def _synthetic_train_tuple(seed: int = 0):
    """Construct a 10-tuple matching the shape produced by
    Datasets.split_dataset, with deterministic synthetic features."""
    g = torch.Generator().manual_seed(seed)
    n_b, n_g = 200, 200
    bdecay_dE = torch.rand(n_b, generator=g) * 0.5 + 0.01
    bdecay_angle = torch.rand(n_b, generator=g) * 2.0 - 1.0
    bdecay_arm = torch.rand(n_b, generator=g) * 0.05 + 0.001
    bg_dE = torch.rand(n_g, generator=g) * 5.0 + 1.0
    bg_angle = torch.rand(n_g, generator=g) * 2.0 - 1.0
    bg_arm = torch.rand(n_g, generator=g) * 1.0 + 0.05

    combined_dE = torch.cat([bdecay_dE, bg_dE])
    combined_angle = torch.cat([bdecay_angle, bg_angle])
    combined_arm = torch.cat([bdecay_arm, bg_arm])
    gt_train = torch.cat(
        [torch.ones(n_b, dtype=torch.bool), torch.zeros(n_g, dtype=torch.bool)]
    )
    return (
        gt_train,
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


def _empty_cfg():
    return OmegaConf.create(
        {
            "likelihoods": {
                "arm": {
                    "sigma_x": 4.0,
                    "n_sigma_cos_theta_kin": 1,
                }
            }
        }
    )


# ---------------------------------------------------------------------------
# Synthetic-data unit tests at the default n_bins=200
# ---------------------------------------------------------------------------


def test_fit_returns_self(wandb_run):
    train = _synthetic_train_tuple()
    trainer = Trainer(_empty_cfg())
    out = trainer.fit(train)
    assert out is trainer


def test_fit_populates_all_required_attributes(wandb_run):
    """After fit(), the trainer must expose bin edges, marginals, conditionals,
    and per-class counts — everything the Evaluator consumes."""
    trainer = Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    expected_attrs = [
        "deltaE_angle_bins",
        "angle_bins",
        "deltaE_arm_bins",
        "arm_bins",
        "bdecay_marginal_deltaE",
        "bg_marginal_deltaE",
        "bdecay_cond_angle",
        "bdecay_cond_arm",
        "bg_cond_angle",
        "bg_cond_arm",
        "n_beta_decay",
        "n_bg",
    ]
    for attr in expected_attrs:
        assert hasattr(trainer, attr), f"Trainer missing attribute {attr!r}"


def test_fit_marginals_sum_to_one(wandb_run):
    """ΔE marginals are 1D probability vectors and must sum to 1."""
    trainer = Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    assert torch.isclose(trainer.bdecay_marginal_deltaE.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(trainer.bg_marginal_deltaE.sum(), torch.tensor(1.0), atol=1e-5)


def test_fit_conditionals_rows_sum_to_one_or_zero(wandb_run):
    """Each row of a conditional matrix is P(y|x_i) for a fixed x bin. The
    row sums to 1 if x_i has any joint mass, else 0 (empty bins are zeroed
    out by ``conditional_from_joint``)."""
    trainer = Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    for matrix in [
        trainer.bdecay_cond_angle,
        trainer.bdecay_cond_arm,
        trainer.bg_cond_angle,
        trainer.bg_cond_arm,
    ]:
        row_sums = matrix.sum(dim=1)
        ones_or_zeros = torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-5) | (
            row_sums == 0.0
        )
        assert torch.all(ones_or_zeros), (
            "row sums of a conditional matrix must be 1 or 0; found "
            f"{row_sums[~ones_or_zeros][:5].tolist()}"
        )


def test_fit_n_counts_match_finite_event_count(wandb_run):
    train = _synthetic_train_tuple()
    trainer = Trainer(_empty_cfg()).fit(train)
    assert trainer.n_beta_decay == int(torch.isfinite(train[1]).sum().item())
    assert trainer.n_bg == int(torch.isfinite(train[4]).sum().item())


def test_fit_no_nan_in_fitted_matrices(wandb_run):
    trainer = Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    for name in [
        "bdecay_marginal_deltaE",
        "bg_marginal_deltaE",
        "bdecay_cond_angle",
        "bdecay_cond_arm",
        "bg_cond_angle",
        "bg_cond_arm",
    ]:
        m = getattr(trainer, name)
        assert torch.all(torch.isfinite(m)), f"{name} contains non-finite values"


def test_fit_matrix_shapes_match_default_n_bins(wandb_run):
    """Production default is 200×200 — verify the conditionals are that size."""
    trainer = Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    assert trainer.bdecay_cond_angle.shape == (200, 200)
    assert trainer.bg_cond_arm.shape == (200, 200)
    assert trainer.bdecay_marginal_deltaE.shape == (200,)


def test_fit_logs_train_split_metrics_to_wandb(wandb_run):
    """Trainer.fit() calls Evaluator(self).evaluate(train, split_name='train')
    internally — that should log the standard metrics."""
    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        Trainer(_empty_cfg()).fit(_synthetic_train_tuple())
    finally:
        wandb.log = real_log
    for key in ["train/TP", "train/FP", "train/FN", "train/TN", "train/precision", "train/recall", "train/f1_score"]:
        assert key in captured, f"Trainer.fit did not log {key!r}"


def test_fit_handles_minimal_data_without_crash(wandb_run):
    """Even with only a handful of events per class, fit() must complete."""
    g = torch.Generator().manual_seed(0)
    n = 20
    train = (
        torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)]),
        torch.rand(n, generator=g) * 0.5 + 0.01,
        torch.rand(n, generator=g) * 2.0 - 1.0,
        torch.rand(n, generator=g) * 0.05 + 0.001,
        torch.rand(n, generator=g) * 5.0 + 1.0,
        torch.rand(n, generator=g) * 2.0 - 1.0,
        torch.rand(n, generator=g) * 1.0 + 0.05,
        torch.cat([torch.rand(n, generator=g) * 0.5 + 0.01, torch.rand(n, generator=g) * 5.0 + 1.0]),
        torch.cat([torch.rand(n, generator=g) * 2.0 - 1.0, torch.rand(n, generator=g) * 2.0 - 1.0]),
        torch.cat([torch.rand(n, generator=g) * 0.05 + 0.001, torch.rand(n, generator=g) * 1.0 + 0.05]),
    )
    trainer = Trainer(_empty_cfg()).fit(train)
    assert trainer.n_beta_decay == n
    assert trainer.n_bg == n


# ---------------------------------------------------------------------------
# Real-data parametrized tests over BIN_SIZES
# ---------------------------------------------------------------------------


def test_real_fit_at_each_bin_size_succeeds(real_trainer):
    """Smoke test — Trainer.fit() at 20×20, 200×200, 1000×1000 must
    complete without error and populate the expected attributes."""
    trainer = real_trainer["trainer"]
    n_bins = real_trainer["n_bins"]
    assert trainer.bdecay_cond_angle.shape == (n_bins, n_bins)
    assert trainer.bg_cond_arm.shape == (n_bins, n_bins)
    assert trainer.bdecay_marginal_deltaE.shape == (n_bins,)


def test_real_fit_n_counts_match_train_split(real_trainer, train_eval_split):
    """The trainer's n_beta_decay / n_bg must equal the finite-event count
    in the train fold."""
    train_tuple, _ = train_eval_split
    trainer = real_trainer["trainer"]
    assert trainer.n_beta_decay == int(torch.isfinite(train_tuple[1]).sum().item())
    assert trainer.n_bg == int(torch.isfinite(train_tuple[4]).sum().item())


def test_real_fit_marginals_sum_to_one(real_trainer):
    trainer = real_trainer["trainer"]
    assert torch.isclose(trainer.bdecay_marginal_deltaE.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(trainer.bg_marginal_deltaE.sum(), torch.tensor(1.0), atol=1e-4)


def test_real_fit_conditionals_rows_sum_to_one_or_zero(real_trainer):
    trainer = real_trainer["trainer"]
    for matrix in [
        trainer.bdecay_cond_angle,
        trainer.bdecay_cond_arm,
        trainer.bg_cond_angle,
        trainer.bg_cond_arm,
    ]:
        row_sums = matrix.sum(dim=1)
        ok = torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-4) | (row_sums == 0.0)
        assert torch.all(ok)


def test_real_fit_no_nan_or_inf_in_fitted_matrices(real_trainer):
    trainer = real_trainer["trainer"]
    for name in [
        "bdecay_marginal_deltaE",
        "bg_marginal_deltaE",
        "bdecay_cond_angle",
        "bdecay_cond_arm",
        "bg_cond_angle",
        "bg_cond_arm",
    ]:
        m = getattr(trainer, name)
        assert torch.all(torch.isfinite(m)), f"{name} has non-finite values at n_bins={real_trainer['n_bins']}"


def test_real_fit_marginal_distribution_visualisation(real_trainer, wandb_run):
    """Plot the per-class ΔE marginal at this bin resolution."""
    trainer = real_trainer["trainer"]
    n_bins = real_trainer["n_bins"]
    edges = trainer.deltaE_angle_bins.cpu().numpy()
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, trainer.bdecay_marginal_deltaE.cpu().numpy(), lw=2, label="β-decay")
    ax.plot(centers, trainer.bg_marginal_deltaE.cpu().numpy(), lw=2, label="background")
    ax.set_xscale("log")
    ax.set_xlabel("ΔE (keV)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Fitted ΔE marginals (n_bins={n_bins})")
    ax.legend()

    wandb_run.log({f"trainer/n_bins={n_bins}/marginal_deltaE": wandb.Image(fig)})
    plt.close(fig)


def test_real_fit_conditional_heatmap_visualisation(real_trainer, wandb_run):
    """Heatmap plot of the four conditional matrices side-by-side at this
    bin resolution."""
    trainer = real_trainer["trainer"]
    n_bins = real_trainer["n_bins"]

    pairs = [
        ("β: angle | ΔE", trainer.bdecay_cond_angle, trainer.deltaE_angle_bins, trainer.angle_bins, False),
        ("bg: angle | ΔE", trainer.bg_cond_angle, trainer.deltaE_angle_bins, trainer.angle_bins, False),
        ("β: ARM | ΔE", trainer.bdecay_cond_arm, trainer.deltaE_arm_bins, trainer.arm_bins, True),
        ("bg: ARM | ΔE", trainer.bg_cond_arm, trainer.deltaE_arm_bins, trainer.arm_bins, True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, matrix, x_bins, y_bins, log_y) in zip(axes.flat, pairs):
        data = matrix.cpu().numpy().T
        vmin = data[data > 0].min() if (data > 0).any() else 1e-12
        norm = mcolors.LogNorm(vmin=vmin, vmax=max(data.max(), vmin * 10))
        im = ax.pcolormesh(
            x_bins.cpu().numpy(),
            y_bins.cpu().numpy(),
            data,
            cmap="viridis",
            norm=norm,
            rasterized=True,
            linewidth=0,
            edgecolors="none",
        )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("ΔE (keV)")
        ax.set_ylabel(title.split(": ")[1])
        ax.set_title(f"{title} (n_bins={n_bins})")
        plt.colorbar(im, ax=ax, label="P(y|ΔE)")

    wandb_run.log({f"trainer/n_bins={n_bins}/conditional_heatmaps": wandb.Image(fig)})
    plt.close(fig)


def test_real_fit_train_f1_recovered_from_wandb(real_trainer, wandb_run):
    """Trainer.fit() logs train/f1_score via the internal Evaluator call.
    Re-evaluate manually under a unique split_name to capture per-bin-size
    train F1, then assert it beats the prevalence baseline."""
    n_bins = real_trainer["n_bins"]
    trainer = real_trainer["trainer"]

    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        from pipeline.eval import Evaluator

        Evaluator(trainer).evaluate(real_trainer["train_tuple"], split_name=f"trainer_self_n_bins={n_bins}")
    finally:
        wandb.log = real_log

    f1_key = f"trainer_self_n_bins={n_bins}/f1_score"
    f1 = captured[f1_key]

    gt = real_trainer["train_tuple"][0]
    prevalence = float(gt.float().mean().item())

    wandb_run.log({f"trainer/n_bins={n_bins}/self_train_f1": f1})
    assert f1 > prevalence, (
        f"At n_bins={n_bins}: self-train F1 ({f1:.3f}) did not beat prevalence baseline ({prevalence:.3f})"
    )
