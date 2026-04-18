"""
Session-scoped fixtures for tests_specialized.

Loads Activation.sim once through MEGAlib/ROOT, then builds the Bayesian
classifier at several density-matrix resolutions so the diagnostic tests can
run at each resolution in turn. The parametrization over n_bins exposes how
much of the classifier's apparent skill comes from sparse-bin memorization
vs. real feature separation.

Requirements:
    - MEGALIB environment variable points to a MEGAlib installation.
    - data/Activation.sim exists relative to the project root.
"""

import os

import pytest
import torch
import wandb
from omegaconf import OmegaConf
from physics.annihilation_detection_utils import (
    R,
    compute_event_features,
    create_tensors,
)
from physics.bayesian_annihilation import BayesianAnnihiliationModel
from physics.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
    lookup_density_values,
    lookup_density_values_1d,
)
from utils.reader_extraction import get_reader

# Resolutions swept by real_posterior_scores. 20×20 is coarse enough that
# every bin is well-populated, 2000×2000 is the production default.
BIN_SIZES = [20, 200, 2000]


@pytest.fixture(scope="session")
def real_tensors():
    """Iterate Activation.sim once and return per-event feature tensors plus
    ground truths. Session-scoped so MEGAlib opens the file exactly once per
    test run, even though several parametrized fixtures depend on it."""
    megalib_root = os.environ["MEGALIB"]
    geo_file = f"{megalib_root}/resource/examples/geomega/special/Max.geo.setup"
    sim_file = "data/Activation.sim"

    cfg = OmegaConf.create(
        {
            "likelihoods": {
                "arm": {
                    "sigma_x": 4.0,
                    "n_sigma_cos_theta_kin": 1,
                }
            }
        }
    )

    reader = get_reader(geo_file, sim_file)

    (
        ground_truths,
        bdecay_list_dE,
        bdecay_list_angle,
        bdecay_list_arm,
        bg_list_dE,
        bg_list_angle,
        bg_list_arm,
        gen_list_dE,
        gen_list_angle,
        gen_list_arm,
    ) = compute_event_features(cfg, ref_energy=511, reader=reader)

    (
        bdecay_dE,
        bdecay_angle,
        bdecay_arm,
        bg_dE,
        bg_angle,
        bg_arm,
        gen_dE,
        gen_angle,
        gen_arm,
        combined_dE,
        combined_angle,
        combined_arm,
    ) = create_tensors(
        bdecay_list_dE,
        bdecay_list_angle,
        bdecay_list_arm,
        bg_list_dE,
        bg_list_angle,
        bg_list_arm,
        gen_list_dE,
        gen_list_angle,
        gen_list_arm,
    )

    return {
        "ground_truths": ground_truths,
        "bdecay_dE": bdecay_dE,
        "bdecay_angle": bdecay_angle,
        "bdecay_arm": bdecay_arm,
        "bg_dE": bg_dE,
        "bg_angle": bg_angle,
        "bg_arm": bg_arm,
        "gen_dE": gen_dE,
        "gen_angle": gen_angle,
        "gen_arm": gen_arm,
        "combined_dE": combined_dE,
        "combined_angle": combined_angle,
        "combined_arm": combined_arm,
    }


def _build_density_matrices_at(t, n_bins):
    """Replicates the production build_density_matrices logic but with a
    configurable bin count on both axes. Kept in the conftest so production
    code stays untouched."""
    # ΔE and ARM are residual-like (peak at 0, long tail) → log spacing.
    # annihilation_angle is a bounded cosine → linear.
    _, deltaE_angle_bins, angle_bins = build_density_matrix(
        t["combined_dE"],
        t["combined_angle"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )
    _, deltaE_arm_bins, arm_bins = build_density_matrix(
        t["combined_dE"],
        t["combined_arm"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=0.1,
        log_y_floor=1e-3,
    )

    bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
        t["bdecay_dE"], t["bdecay_angle"], x_bins=deltaE_angle_bins, y_bins=angle_bins
    )
    bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
        t["bdecay_dE"], t["bdecay_arm"], x_bins=deltaE_arm_bins, y_bins=arm_bins
    )
    bg_joint_deltaE_angle, _, _ = build_density_matrix(
        t["bg_dE"], t["bg_angle"], x_bins=deltaE_angle_bins, y_bins=angle_bins
    )
    bg_joint_deltaE_arm, _, _ = build_density_matrix(
        t["bg_dE"], t["bg_arm"], x_bins=deltaE_arm_bins, y_bins=arm_bins
    )

    bdecay_marginal_dE, _ = build_density_matrix_1d(t["bdecay_dE"], x_bins=deltaE_angle_bins)
    bg_marginal_dE, _ = build_density_matrix_1d(t["bg_dE"], x_bins=deltaE_angle_bins)

    bdecay_cond_angle = conditional_from_joint(bdecay_joint_deltaE_angle, axis=0)
    bdecay_cond_arm = conditional_from_joint(bdecay_joint_deltaE_arm, axis=0)
    bg_cond_angle = conditional_from_joint(bg_joint_deltaE_angle, axis=0)
    bg_cond_arm = conditional_from_joint(bg_joint_deltaE_arm, axis=0)

    p_beta_deltaE = lookup_density_values_1d(t["gen_dE"], bdecay_marginal_dE, deltaE_angle_bins)
    p_bg_deltaE = lookup_density_values_1d(t["gen_dE"], bg_marginal_dE, deltaE_angle_bins)
    p_beta_angle = lookup_density_values(
        t["gen_dE"], t["gen_angle"], bdecay_cond_angle, deltaE_angle_bins, angle_bins
    )
    p_bg_angle = lookup_density_values(
        t["gen_dE"], t["gen_angle"], bg_cond_angle, deltaE_angle_bins, angle_bins
    )
    p_beta_arm = lookup_density_values(t["gen_dE"], t["gen_arm"], bdecay_cond_arm, deltaE_arm_bins, arm_bins)
    p_bg_arm = lookup_density_values(t["gen_dE"], t["gen_arm"], bg_cond_arm, deltaE_arm_bins, arm_bins)

    return (p_beta_deltaE, p_bg_deltaE, p_beta_angle, p_bg_angle, p_beta_arm, p_bg_arm)


@pytest.fixture(scope="session", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def real_posterior_scores(real_tensors, request):
    """Build the Bayesian classifier at resolution ``request.param`` and
    return the model, ground-truth labels, valid-event mask, and bin count.
    Parametrized over BIN_SIZES so every diagnostic test runs once per
    resolution."""
    n_bins = request.param

    (
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle,
        p_bg_angle,
        p_beta_arm,
        p_bg_arm,
    ) = _build_density_matrices_at(real_tensors, n_bins)

    n_beta_decay = int(torch.isfinite(real_tensors["bdecay_dE"]).sum().item())
    n_bg = int(torch.isfinite(real_tensors["bg_dE"]).sum().item())

    ratio = R(
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle,
        p_bg_angle,
        p_beta_arm,
        p_bg_arm,
    )
    model = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg)

    valid = torch.isfinite(ratio).numpy()
    labels = torch.tensor(real_tensors["ground_truths"], dtype=torch.bool).numpy().astype("int64")

    return {"model": model, "labels": labels, "valid": valid, "n_bins": n_bins}


@pytest.fixture(scope="session")
def wandb_run():
    """Single W&B run collecting diagnostics across every bin-size parametrization."""
    run = wandb.init(
        project="cosi-betadecay-tests",
        name="test-classifier-diagnostics",
        tags=["tests", "classifier", "diagnostics"],
    )
    yield run
    wandb.finish()
