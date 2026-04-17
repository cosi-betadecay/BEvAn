import os

import pytest
import torch
import wandb
from omegaconf import OmegaConf
from physics.annihilation_detection_utils import (
    R,
    build_density_matrices,
    compute_event_features,
    create_tensors,
)
from physics.bayesian_annihilation import BayesianAnnihiliationModel
from utils.reader_extraction import get_reader


@pytest.fixture(scope="session")
def real_posterior_scores():
    """
    Load data/Activation.sim, run the Bayesian pipeline, and return:

        model   : BayesianAnnihiliationModel instance (call its methods for
                  P(β|D), P(bg|D), predictions, etc.)
        labels  : int64 ground-truth labels (1 = β, 0 = bg)
        valid   : boolean mask (1D, length = n_events) selecting events that
                  survived event_data_processing. Apply this mask to whatever
                  the model returns before pairing with `labels`.
    """
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
        _combined_dE,
        _combined_angle,
        _combined_arm,
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

    (
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
    ) = build_density_matrices(
        bdecay_dE,
        bdecay_angle,
        bdecay_arm,
        bg_dE,
        bg_angle,
        bg_arm,
        gen_dE,
        gen_angle,
        gen_arm,
        torch.cat([bdecay_dE, bg_dE]),
        torch.cat([bdecay_angle, bg_angle]),
        torch.cat([bdecay_arm, bg_arm]),
    )

    n_beta_decay = int(torch.isfinite(bdecay_dE).sum().item())
    n_bg = int(torch.isfinite(bg_dE).sum().item())

    ratio = R(
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
    )

    model = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg)

    # Events dropped by event_data_processing propagate NaN through R → ratio.
    # Compute a reusable validity mask from the ratio itself.
    valid = torch.isfinite(ratio).numpy()
    labels = torch.tensor(ground_truths, dtype=torch.bool).numpy().astype("int64")

    return {"model": model, "labels": labels, "valid": valid}


@pytest.fixture(scope="session")
def wandb_run():
    """Single W&B run for the diagnostic tests; logs when online, harmless when
    offline."""
    run = wandb.init(
        project="cosi-betadecay-tests",
        name="test-classifier-diagnostics",
        tags=["tests", "classifier", "diagnostics"],
    )
    yield run
    wandb.finish()
