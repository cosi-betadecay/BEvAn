"""
Session-scoped pytest fixtures shared across tests_physics.

The real_tensors fixture processes the full Activation.sim file through the
same event-reading and feature-extraction pipeline that production code uses.
Because it is session-scoped, MEGAlib opens and iterates the file only once
per test run, regardless of how many test functions consume the fixture.

Requirements:
    - The MEGALIB environment variable must point to a valid MEGAlib installation.
    - data/Activation.sim must exist relative to the project root (where pytest
      is invoked from).
"""

import os

import pytest
import torch
import wandb
from omegaconf import OmegaConf

from physics.annihilation_detection_utils import compute_event_features, create_tensors
from physics.matrix_calculations import build_density_matrix
from utils.reader_extraction import get_reader

# Resolution used for the pre-built test matrices.
# 200×200 is sufficient for all quantitative assertions and runs in seconds.
# Production uses 2000×2000; increase here only if a test demands finer bins.
N_BINS_TEST = 200


@pytest.fixture(scope="session")
def real_tensors():
    """
    Load and process the full Activation.sim file, returning every feature
    tensor that the pipeline produces.

    The fixture mirrors the first half of annihilation_extractor() exactly:
        1. Initialize the MEGAlib reader with the production geometry file.
        2. Call compute_event_features() to iterate over all events and
           produce per-class feature lists (delta_E, annihilation_angle, ARM).
        3. Call create_tensors() to convert those lists into PyTorch tensors
           and build the combined (bdecay + bg) tensors.

    Returned dict keys
    ------------------
    ground_truths : torch.BoolTensor, shape (n_events,)
        True for beta-decay events, False for background, in event order.
    bdecay_tensor_delta_E : Tensor  — delta_E values for bdecay events only
    bdecay_tensor_annihilation_angle : Tensor
    bdecay_tensor_arm : Tensor
    bg_tensor_delta_E : Tensor      — delta_E values for background events only
    bg_tensor_annihilation_angle : Tensor
    bg_tensor_arm : Tensor
    gen_tensor_delta_E : Tensor     — all events in original event order
    gen_tensor_annihilation_angle : Tensor
    gen_tensor_arm : Tensor
    combined_tensor_delta_E : Tensor  — cat([bdecay, bg])
    combined_tensor_annihilation_angle : Tensor
    combined_tensor_arm : Tensor
    n_gen : int  — total number of events (== len(gen_tensor_*))
    """
    megalib_root = os.environ["MEGALIB"]
    geo_file = f"{megalib_root}/resource/examples/geomega/special/Max.geo.setup"
    sim_file = "data/Activation.sim"

    # ARM parameters from configs/likelihoods.yaml
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
        "ground_truths": torch.tensor(ground_truths, dtype=torch.bool),
        "bdecay_tensor_delta_E": bdecay_dE,
        "bdecay_tensor_annihilation_angle": bdecay_angle,
        "bdecay_tensor_arm": bdecay_arm,
        "bg_tensor_delta_E": bg_dE,
        "bg_tensor_annihilation_angle": bg_angle,
        "bg_tensor_arm": bg_arm,
        "gen_tensor_delta_E": gen_dE,
        "gen_tensor_annihilation_angle": gen_angle,
        "gen_tensor_arm": gen_arm,
        "combined_tensor_delta_E": combined_dE,
        "combined_tensor_annihilation_angle": combined_angle,
        "combined_tensor_arm": combined_arm,
        "n_gen": gen_dE.numel(),
    }


@pytest.fixture(scope="session")
def wandb_run():
    """
    Initialize a single Weights & Biases run for the entire test session.

    All test plots are logged to this one run so that every heatmap,
    marginal overlay, and scatter plot can be compared side-by-side inside
    the W&B dashboard without switching between runs.

    The run is finalized automatically via wandb.finish() at session teardown.
    """
    run = wandb.init(
        project="cosi-betadecay-tests",
        name="test-density-matrices",
        tags=["tests", "matrix", "density"],
    )
    yield run
    wandb.finish()


@pytest.fixture(scope="session")
def density_matrices(real_tensors):
    """
    Pre-built 2D probability matrices for quantitative test assertions.

    Both feature pairs are built with N_BINS_TEST × N_BINS_TEST resolution
    using shared bin edges derived from the combined (bdecay + bg) tensor —
    exactly mirroring the production pipeline in build_density_matrices().

    Returned dict keys
    ------------------
    bdecay_angle  : (N, N) probability matrix — bdecay, delta_E vs angle
    bg_angle      : (N, N) probability matrix — background, delta_E vs angle
    x_bins_angle  : (N+1,) shared delta_E bin edges for the angle pair
    y_bins_angle  : (N+1,) shared annihilation-angle bin edges
    bdecay_arm    : (N, N) probability matrix — bdecay, delta_E vs ARM
    bg_arm        : (N, N) probability matrix — background, delta_E vs ARM
    x_bins_arm    : (N+1,) shared delta_E bin edges for the ARM pair
    y_bins_arm    : (N+1,) shared ARM bin edges
    """
    t = real_tensors

    # ---- Angle feature pair: delta_E vs annihilation_angle ----
    _, x_bins_angle, y_bins_angle = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        n_bins_x=N_BINS_TEST,
        n_bins_y=N_BINS_TEST,
    )
    bdecay_angle, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )
    bg_angle, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )

    # ---- ARM feature pair: delta_E vs ARM ----
    _, x_bins_arm, y_bins_arm = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_arm"],
        n_bins_x=N_BINS_TEST,
        n_bins_y=N_BINS_TEST,
    )
    bdecay_arm, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )
    bg_arm, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )

    return {
        "bdecay_angle": bdecay_angle,
        "bg_angle": bg_angle,
        "x_bins_angle": x_bins_angle,
        "y_bins_angle": y_bins_angle,
        "bdecay_arm": bdecay_arm,
        "bg_arm": bg_arm,
        "x_bins_arm": x_bins_arm,
        "y_bins_arm": y_bins_arm,
    }
