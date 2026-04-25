import os

import pytest
import torch
import wandb
from dataset.datasets import Datasets
from modeling.bayesian_annihilation import BayesianAnnihiliationModel
from modeling.calculate_probablities import R
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
    lookup_density_values,
    lookup_density_values_1d,
)
from omegaconf import OmegaConf
from physics.compton_cone_reconstruction import FarFieldImager
from utils.reader_extraction import get_reader

# Resolutions swept by real_posterior_scores. 20×20 is coarse enough that
# every bin is well-populated, 1000×1000 is the production default. 300×300
# bridges the two so the curse-of-dimensionality transition is visible.
BIN_SIZES = [20, 300, 1000]


@pytest.fixture(scope="session")
def real_tensors():
    """Iterate Activation.sim once and return per-event feature tensors plus
    ground truths. Session-scoped so MEGAlib opens the file exactly once per
    test run, even though several parametrized fixtures depend on it."""
    megalib_root = os.environ["MEGALIB"]
    geo_file = f"{megalib_root}/resource/examples/geomega/special/Max.geo.setup"
    sim_file = "data/Activation.sim"
    tra_file = "data/Activation.tra"

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

    # Replicates the mainline pipeline in run.py: run the far-field
    # backprojection once per session so ARM has a real reconstructed source
    # direction instead of a hard-coded axis.
    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    imager.backproject_file(tra_file)
    reconstructed_unit_vector = imager.peak_direction_cartesian()

    reader_sim, reader_tra = get_reader(geo_file, sim_file)
    datasets = Datasets(cfg, 511, reader_sim, reader_tra, reconstructed_unit_vector)

    (
        ground_truths,
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
    ) = datasets.compute_event_features(cfg, 511, reader_sim, reader_tra, reconstructed_unit_vector)

    return {
        # ``datasets`` is exposed so downstream fixtures (train_eval_split)
        # can call ``datasets.split_dataset(...)`` — the same entry point
        # run.py uses before training.
        "datasets": datasets,
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


@pytest.fixture(scope="session")
def train_eval_split(real_tensors):
    """80/20 stratified split matching what run.py does before training.

    Returns ``(train_tuple, eval_tuple)`` in the 10-tuple layout produced by
    ``Datasets.split_dataset`` — same call the production pipeline makes, so
    the held-out eval set here is the same one the Trainer never sees.
    """
    datasets = real_tensors["datasets"]
    return datasets.split_dataset(
        real_tensors["bdecay_dE"],
        real_tensors["bdecay_angle"],
        real_tensors["bdecay_arm"],
        real_tensors["bg_dE"],
        real_tensors["bg_angle"],
        real_tensors["bg_arm"],
    )


def _unpack_split_tuple(tup):
    """Unpack the 10-tuple ``split_dataset`` returns into a dict keyed the
    same way ``real_tensors`` is — lets the build/score helper accept either
    the train half or the eval half without branching."""
    (
        ground_truths,
        bdecay_dE,
        bdecay_angle,
        bdecay_arm,
        bg_dE,
        bg_angle,
        bg_arm,
        combined_dE,
        combined_angle,
        combined_arm,
    ) = tup
    return {
        "ground_truths": ground_truths,
        "bdecay_dE": bdecay_dE,
        "bdecay_angle": bdecay_angle,
        "bdecay_arm": bdecay_arm,
        "bg_dE": bg_dE,
        "bg_angle": bg_angle,
        "bg_arm": bg_arm,
        "combined_dE": combined_dE,
        "combined_angle": combined_angle,
        "combined_arm": combined_arm,
    }


def _build_density_matrices_split(train_t, eval_t, n_bins):
    """Mirrors ``_build_density_matrices_at`` but with a strict train/eval
    separation: density matrices and bin edges are derived from the train
    tensors only, then the eval combined-tensors are fed into the lookups to
    produce per-event likelihoods for the held-out split.

    This is the difference the user caught — ``_build_density_matrices_at``
    uses ``t["gen_dE"]`` (all events) for both the build and the lookup, so
    the resulting ROC/PR is a train-set fit curve. Here the lookup inputs
    come from eval only, so the curve reflects generalization.
    """
    _, deltaE_angle_bins, angle_bins = build_density_matrix(
        train_t["combined_dE"],
        train_t["combined_angle"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )
    _, deltaE_arm_bins, arm_bins = build_density_matrix(
        train_t["combined_dE"],
        train_t["combined_arm"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=0.1,
        log_y_floor=1e-3,
    )

    bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
        train_t["bdecay_dE"], train_t["bdecay_angle"], x_bins=deltaE_angle_bins, y_bins=angle_bins
    )
    bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
        train_t["bdecay_dE"], train_t["bdecay_arm"], x_bins=deltaE_arm_bins, y_bins=arm_bins
    )
    bg_joint_deltaE_angle, _, _ = build_density_matrix(
        train_t["bg_dE"], train_t["bg_angle"], x_bins=deltaE_angle_bins, y_bins=angle_bins
    )
    bg_joint_deltaE_arm, _, _ = build_density_matrix(
        train_t["bg_dE"], train_t["bg_arm"], x_bins=deltaE_arm_bins, y_bins=arm_bins
    )

    bdecay_marginal_dE, _ = build_density_matrix_1d(train_t["bdecay_dE"], x_bins=deltaE_angle_bins)
    bg_marginal_dE, _ = build_density_matrix_1d(train_t["bg_dE"], x_bins=deltaE_angle_bins)

    bdecay_cond_angle = conditional_from_joint(bdecay_joint_deltaE_angle, axis=0)
    bdecay_cond_arm = conditional_from_joint(bdecay_joint_deltaE_arm, axis=0)
    bg_cond_angle = conditional_from_joint(bg_joint_deltaE_angle, axis=0)
    bg_cond_arm = conditional_from_joint(bg_joint_deltaE_arm, axis=0)

    # Lookups consume eval combined tensors. ``combined_dE`` in a split tuple
    # is ``cat([bdecay_eval_dE, bg_eval_dE])`` and ``ground_truths`` matches
    # that order, so the per-event ratio lines up with the eval labels.
    p_beta_deltaE = lookup_density_values_1d(eval_t["combined_dE"], bdecay_marginal_dE, deltaE_angle_bins)
    p_bg_deltaE = lookup_density_values_1d(eval_t["combined_dE"], bg_marginal_dE, deltaE_angle_bins)
    p_beta_angle = lookup_density_values(
        eval_t["combined_dE"], eval_t["combined_angle"], bdecay_cond_angle, deltaE_angle_bins, angle_bins
    )
    p_bg_angle = lookup_density_values(
        eval_t["combined_dE"], eval_t["combined_angle"], bg_cond_angle, deltaE_angle_bins, angle_bins
    )
    p_beta_arm = lookup_density_values(
        eval_t["combined_dE"], eval_t["combined_arm"], bdecay_cond_arm, deltaE_arm_bins, arm_bins
    )
    p_bg_arm = lookup_density_values(
        eval_t["combined_dE"], eval_t["combined_arm"], bg_cond_arm, deltaE_arm_bins, arm_bins
    )

    return (p_beta_deltaE, p_bg_deltaE, p_beta_angle, p_bg_angle, p_beta_arm, p_bg_arm)


@pytest.fixture(scope="session", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def eval_posterior_scores(train_eval_split, request):
    """Bayesian classifier at resolution ``request.param``, trained on the 80%
    split and scored on the held-out 20%.

    Prior counts (``n_beta_decay``, ``n_bg``) come from the TRAIN split
    because that's what the classifier legitimately "knows" at inference
    time — using eval counts would leak the label distribution of the
    held-out set into the prior.

    Return shape is intentionally the same as ``real_posterior_scores`` so
    ROC / PR / score-distribution tests can swap in this fixture and run
    unchanged. ``labels``, ``valid``, and model outputs are all sized to the
    eval split (~20% of total events).
    """
    n_bins = request.param
    train_tup, eval_tup = train_eval_split
    train_t = _unpack_split_tuple(train_tup)
    eval_t = _unpack_split_tuple(eval_tup)

    (
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle,
        p_bg_angle,
        p_beta_arm,
        p_bg_arm,
    ) = _build_density_matrices_split(train_t, eval_t, n_bins)

    n_beta_decay = int(torch.isfinite(train_t["bdecay_dE"]).sum().item())
    n_bg = int(torch.isfinite(train_t["bg_dE"]).sum().item())

    ratio = R(p_beta_deltaE, p_bg_deltaE, p_beta_angle, p_bg_angle, p_beta_arm, p_bg_arm)
    model = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg)

    valid = torch.isfinite(ratio).numpy()
    labels = eval_t["ground_truths"].cpu().numpy().astype("int64")

    return {
        "model": model,
        "labels": labels,
        "valid": valid,
        "n_bins": n_bins,
        "n_beta_decay": n_beta_decay,
        "n_bg": n_bg,
        "p_beta_deltaE": p_beta_deltaE,
        "p_bg_deltaE": p_bg_deltaE,
        "p_beta_angle": p_beta_angle,
        "p_bg_angle": p_bg_angle,
        "p_beta_arm": p_beta_arm,
        "p_bg_arm": p_bg_arm,
        "split": "eval",
    }


@pytest.fixture(scope="session", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def real_posterior_scores(real_tensors, request):
    """Build the Bayesian classifier at resolution ``request.param`` and
    return the model, ground-truth labels, valid-event mask, bin count, prior
    counts, and the per-feature per-event likelihood lookups so downstream
    tests can ablate individual feature contributions without rebuilding
    the matrices.

    Note: this fixture uses the full event set for both the density-matrix
    build and the lookup, so metrics derived from it reflect train-set fit,
    not generalization. Use ``eval_posterior_scores`` for held-out metrics.
    """
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

    return {
        "model": model,
        "labels": labels,
        "valid": valid,
        "n_bins": n_bins,
        "n_beta_decay": n_beta_decay,
        "n_bg": n_bg,
        "p_beta_deltaE": p_beta_deltaE,
        "p_bg_deltaE": p_bg_deltaE,
        "p_beta_angle": p_beta_angle,
        "p_bg_angle": p_bg_angle,
        "p_beta_arm": p_beta_arm,
        "p_bg_arm": p_bg_arm,
    }


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
