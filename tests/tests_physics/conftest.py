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
from omegaconf import OmegaConf

from physics.annihilation_detection_utils import compute_event_features, create_tensors
from utils.reader_extraction import get_reader


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
