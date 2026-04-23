import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from modeling.bayesian_annihilation import BayesianAnnihiliationModel

# ---------------------------------------------------------------------------
# Prior contract
# ---------------------------------------------------------------------------


def test_priors_sum_to_one():
    model = BayesianAnnihiliationModel(torch.tensor([1.0]), n_beta_decay=300, n_bg=700)
    p_beta = model.beta_decay_probability()
    p_bg = model.background_probability()
    assert math.isclose(p_beta + p_bg, 1.0, abs_tol=1e-9)


def test_priors_match_count_fractions():
    model = BayesianAnnihiliationModel(torch.tensor([1.0]), n_beta_decay=200, n_bg=800)
    assert math.isclose(model.beta_decay_probability(), 0.2, abs_tol=1e-9)
    assert math.isclose(model.background_probability(), 0.8, abs_tol=1e-9)


def test_priors_handle_balanced_classes():
    model = BayesianAnnihiliationModel(torch.tensor([1.0]), n_beta_decay=500, n_bg=500)
    assert math.isclose(model.beta_decay_probability(), 0.5, abs_tol=1e-9)
    assert math.isclose(model.background_probability(), 0.5, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Likelihood contract: P(D|β) + P(D|bg) = 1 (ratio-derived parameterisation)
# ---------------------------------------------------------------------------


def test_data_given_class_probabilities_sum_to_one():
    """The class re-parameterises P(D|·) from the ratio: P(D|β)=R/(R+1),
    P(D|bg)=1/(R+1). They must sum to 1 elementwise."""
    R = torch.tensor([0.1, 1.0, 5.0, 100.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1)
    p_beta = model.data_given_beta_decay_probability()
    p_bg = model.data_given_background_probability()
    assert torch.allclose(p_beta + p_bg, torch.ones_like(p_beta), atol=1e-9)


def test_data_given_class_in_unit_interval():
    R = torch.tensor([1e-6, 1e-3, 1.0, 1e3, 1e6])
    model = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1)
    p_beta = model.data_given_beta_decay_probability()
    p_bg = model.data_given_background_probability()
    assert torch.all(p_beta >= 0.0) and torch.all(p_beta <= 1.0)
    assert torch.all(p_bg >= 0.0) and torch.all(p_bg <= 1.0)


def test_data_given_beta_monotonic_in_ratio():
    """As R grows, P(D|β) strictly grows (Bayes ratio is monotone)."""
    R = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
    p_beta = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1).data_given_beta_decay_probability()
    assert torch.all(torch.diff(p_beta) > 0)


# ---------------------------------------------------------------------------
# Posterior contract
# ---------------------------------------------------------------------------


def test_posterior_probabilities_sum_to_one():
    """Bayesian posterior probabilities for a 2-class problem sum to 1."""
    R = torch.tensor([0.1, 1.0, 5.0, 100.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=300, n_bg=700)
    p_beta = model.beta_decay_given_data_probability()
    p_bg = model.background_given_data_probability()
    assert torch.allclose(p_beta + p_bg, torch.ones_like(p_beta), atol=1e-9)


def test_posterior_in_unit_interval():
    R = torch.tensor([1e-6, 1.0, 1e6])
    model = BayesianAnnihiliationModel(R, n_beta_decay=400, n_bg=600)
    p_beta = model.beta_decay_given_data_probability()
    p_bg = model.background_given_data_probability()
    assert torch.all(p_beta >= 0.0) and torch.all(p_beta <= 1.0)
    assert torch.all(p_bg >= 0.0) and torch.all(p_bg <= 1.0)


def test_posterior_at_unit_ratio_equals_prior():
    """When the data is uninformative (R = 1), the posterior collapses to
    the prior."""
    R = torch.tensor([1.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=200, n_bg=800)
    p_beta_post = model.beta_decay_given_data_probability().item()
    p_beta_prior = model.beta_decay_probability()
    assert math.isclose(p_beta_post, p_beta_prior, abs_tol=1e-9)


def test_posterior_with_balanced_prior_collapses_to_normalised_ratio():
    """With equal priors, posterior P(β|D) reduces to R/(R+1)."""
    R = torch.tensor([0.5, 2.0, 10.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1)
    p_beta_post = model.beta_decay_given_data_probability()
    expected = R / (R + 1.0)
    assert torch.allclose(p_beta_post, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------


def test_inference_returns_bool_tensor():
    R = torch.tensor([0.5, 1.0, 2.0])
    out = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1).inference()
    assert out.dtype == torch.bool
    assert out.shape == R.shape


def test_inference_extreme_ratio_predicts_beta():
    """Very large R must always classify as β regardless of the prior."""
    R = torch.tensor([1e10])
    for n_beta, n_bg in [(1, 1), (1, 1000), (500, 500), (100, 999000)]:
        out = BayesianAnnihiliationModel(R, n_beta_decay=n_beta, n_bg=n_bg).inference()
        assert out.item() is True, f"failed at prior n_β={n_beta}, n_bg={n_bg}"


def test_inference_extreme_low_ratio_predicts_bg():
    """Very small R must always classify as bg regardless of the prior."""
    R = torch.tensor([1e-10])
    for n_beta, n_bg in [(1, 1), (999000, 100), (500, 500), (1000, 1)]:
        out = BayesianAnnihiliationModel(R, n_beta_decay=n_beta, n_bg=n_bg).inference()
        assert out.item() is False, f"failed at prior n_β={n_beta}, n_bg={n_bg}"


def test_inference_threshold_at_unit_ratio_with_equal_priors():
    """At the decision boundary (R = 1, balanced prior), inference is True
    (the rule is ≥, not >)."""
    R = torch.tensor([1.0])
    out = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1).inference()
    assert out.item() is True


def test_inference_imbalanced_prior_shifts_threshold():
    """With a strong bg prior, larger R is needed to swing the decision to β.
    Find the crossover R for two priors and confirm the imbalanced one is
    higher."""
    Rs = torch.tensor(np.geomspace(0.01, 100.0, 1000))
    balanced = BayesianAnnihiliationModel(Rs, n_beta_decay=500, n_bg=500).inference()
    imbalanced = BayesianAnnihiliationModel(Rs, n_beta_decay=10, n_bg=990).inference()

    crossover_balanced = Rs[balanced.long().argmax()].item()
    crossover_imbalanced = Rs[imbalanced.long().argmax()].item()
    assert crossover_imbalanced > crossover_balanced, (
        f"imbalanced prior should require higher R to switch (got "
        f"{crossover_imbalanced:.3f} vs balanced {crossover_balanced:.3f})"
    )


def test_inference_consistent_with_posterior_comparison():
    """inference() and (P(β|D) ≥ P(bg|D)) must always agree."""
    R = torch.tensor([0.01, 0.5, 1.0, 2.0, 100.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=300, n_bg=700)
    inf = model.inference()
    direct = model.beta_decay_given_data_probability() >= model.background_given_data_probability()
    assert torch.equal(inf, direct)


# ---------------------------------------------------------------------------
# Boundary / numerical edge cases
# ---------------------------------------------------------------------------


def test_zero_ratio_returns_zero_p_beta_likelihood():
    R = torch.tensor([0.0])
    model = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1)
    assert math.isclose(model.data_given_beta_decay_probability().item(), 0.0, abs_tol=1e-9)
    assert math.isclose(model.data_given_background_probability().item(), 1.0, abs_tol=1e-9)


def test_inference_supports_higher_dim_input():
    """The class must broadcast over arbitrary ratio shapes."""
    R = torch.rand(4, 5)
    out = BayesianAnnihiliationModel(R, n_beta_decay=1, n_bg=1).inference()
    assert out.shape == R.shape


# ---------------------------------------------------------------------------
# Decision-boundary visualisations (W&B-logged)
# ---------------------------------------------------------------------------


def test_posterior_vs_ratio_for_several_priors(wandb_run):
    """Plot posterior P(β|D) against R for a sweep of prior fractions; the
    family of curves should fan smoothly from the all-bg curve at the bottom
    to the all-β curve at the top."""
    R = torch.tensor(np.geomspace(0.01, 100.0, 500))
    fractions = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    fig, ax = plt.subplots(figsize=(7, 5))
    for f in fractions:
        n_beta = int(round(f * 1000))
        n_bg = 1000 - n_beta
        model = BayesianAnnihiliationModel(R, n_beta_decay=n_beta, n_bg=n_bg)
        post = model.beta_decay_given_data_probability().numpy()
        ax.plot(R.numpy(), post, label=f"P(β) = {f:.2f}")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="decision threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Likelihood ratio R = P(D|β) / P(D|bg)")
    ax.set_ylabel("Posterior P(β | D)")
    ax.set_title("Bayes posterior vs likelihood ratio for varying priors")
    ax.legend(loc="lower right", fontsize=8)

    wandb_run.log({"bayes/posterior_vs_ratio_sweep": wandb.Image(fig)})
    plt.close(fig)


def test_decision_boundary_shifts_with_prior(wandb_run):
    """At what R does the model flip from bg to β? Plot the crossover R as
    a function of the prior fraction P(β); for the standard rule this is
    R* = P(bg)/P(β)."""
    fractions = np.linspace(0.05, 0.95, 50)
    Rs = torch.tensor(np.geomspace(1e-3, 1e3, 4000))

    crossovers = []
    expected = []
    for f in fractions:
        n_beta = int(round(f * 10000))
        n_bg = 10000 - n_beta
        model = BayesianAnnihiliationModel(Rs, n_beta_decay=n_beta, n_bg=n_bg)
        decisions = model.inference().long()
        first_true = int(decisions.argmax().item())
        crossovers.append(float(Rs[first_true].item()))
        expected.append((1.0 - f) / max(f, 1e-9))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, crossovers, "o-", label="empirical R*", markersize=4)
    ax.plot(fractions, expected, "--", color="gray", label="theoretical R* = P(bg)/P(β)")
    ax.set_yscale("log")
    ax.set_xlabel("Prior P(β)")
    ax.set_ylabel("Decision-boundary ratio R*")
    ax.set_title("Where the classifier flips bg → β as a function of the prior")
    ax.legend()

    wandb_run.log({"bayes/decision_boundary_vs_prior": wandb.Image(fig)})
    plt.close(fig)

    log_grid_step = math.log10(Rs[1].item() / Rs[0].item())
    log_diff = np.abs(np.log10(np.array(crossovers)) - np.log10(np.array(expected)))
    assert np.all(log_diff <= 5 * log_grid_step), "Empirical decision boundary deviates from R* = P(bg)/P(β)"


def test_posterior_heatmap_over_ratio_and_prior(wandb_run):
    """Heatmap of posterior P(β|D) over the (R, P(β)) plane — the iso-50%
    contour traces the decision boundary."""
    Rs = np.geomspace(1e-3, 1e3, 100)
    fractions = np.linspace(0.02, 0.98, 80)
    grid = np.zeros((fractions.size, Rs.size))
    R_t = torch.tensor(Rs)
    for i, f in enumerate(fractions):
        n_beta = int(round(f * 10000))
        n_bg = 10000 - n_beta
        post = (
            BayesianAnnihiliationModel(R_t, n_beta_decay=n_beta, n_bg=n_bg)
            .beta_decay_given_data_probability()
            .numpy()
        )
        grid[i] = post

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(Rs, fractions, grid, cmap="RdBu_r", vmin=0.0, vmax=1.0, shading="auto")
    ax.contour(Rs, fractions, grid, levels=[0.5], colors="black", linewidths=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("R")
    ax.set_ylabel("Prior P(β)")
    ax.set_title("Posterior P(β | D); 0.5 contour = decision boundary")
    plt.colorbar(im, ax=ax, label="P(β | D)")

    wandb_run.log({"bayes/posterior_heatmap": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Real-data test (uses the parent conftest's real_tensors)
# ---------------------------------------------------------------------------


def test_real_data_posterior_distribution_per_class(real_tensors, wandb_run):
    """Build a synthetic ratio from the real ΔE marginals (P_β(ΔE)/P_bg(ΔE),
    histogram-derived) and plot the posterior distribution per class. The
    true-β population should sit higher than the true-bg population."""
    bdecay_dE = real_tensors["bdecay_tensor_delta_E"]
    bg_dE = real_tensors["bg_tensor_delta_E"]
    gen_dE = real_tensors["gen_tensor_delta_E"]
    gt = real_tensors["ground_truths"]

    bd = bdecay_dE[torch.isfinite(bdecay_dE) & (bdecay_dE > 0)].cpu().numpy()
    bg = bg_dE[torch.isfinite(bg_dE) & (bg_dE > 0)].cpu().numpy()
    if bd.size == 0 or bg.size == 0:
        pytest.skip("no positive ΔE values available")

    edges = np.geomspace(
        max(min(bd.min(), bg.min()), 1e-3),
        max(bd.max(), bg.max(), 1e-2),
        80,
    )
    p_beta_hist, _ = np.histogram(bd, bins=edges, density=True)
    p_bg_hist, _ = np.histogram(bg, bins=edges, density=True)

    gen = gen_dE.cpu().numpy()
    finite = np.isfinite(gen) & (gen > edges[0]) & (gen < edges[-1])
    idx = np.searchsorted(edges, gen[finite], side="right") - 1
    idx = np.clip(idx, 0, p_beta_hist.size - 1)

    eps = 1e-8
    R = (p_beta_hist[idx] + eps) / (p_bg_hist[idx] + eps)
    R_t = torch.tensor(R, dtype=torch.float64)

    n_beta = int(torch.isfinite(bdecay_dE).sum().item())
    n_bg = int(torch.isfinite(bg_dE).sum().item())
    model = BayesianAnnihiliationModel(R_t, n_beta_decay=n_beta, n_bg=n_bg)
    posterior = model.beta_decay_given_data_probability().numpy()

    gt_finite = gt.cpu().numpy()[finite].astype(bool)
    posterior_beta = posterior[gt_finite]
    posterior_bg = posterior[~gt_finite]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(
        posterior_bg,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true bg (n={posterior_bg.size}, μ={posterior_bg.mean():.2f})",
    )
    ax.hist(
        posterior_beta,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true β (n={posterior_beta.size}, μ={posterior_beta.mean():.2f})",
    )
    ax.set_xlabel("Posterior P(β | ΔE only)")
    ax.set_ylabel("density")
    ax.set_title("Real-data ΔE-only Bayes posterior per true class")
    ax.legend()

    wandb_run.log({"bayes/real_data_dE_only_posterior": wandb.Image(fig)})
    plt.close(fig)

    assert posterior_beta.mean() > posterior_bg.mean(), (
        f"true-β posterior mean ({posterior_beta.mean():.3f}) should exceed "
        f"true-bg posterior mean ({posterior_bg.mean():.3f}) on the ΔE-only model"
    )
