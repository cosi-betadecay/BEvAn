import omegaconf
import torch
from utils.reader_extraction import get_reader

from physics.annihilation_detection_utils import (
    build_density_matrices,
    compute_event_features,
    create_tensors,
    predict,
)


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    reconstructed_unit_vector,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    (
        ground_truths,
        bdecay_list_delta_E,
        bdecay_list_annihilation_angle,
        bdecay_list_arm,
        bg_list_delta_E,
        bg_list_annihilation_angle,
        bg_list_arm,
        gen_list_delta_E,
        gen_list_annihilation_angle,
        gen_list_arm,
    ) = compute_event_features(cfg, ref_energy, reader, reconstructed_unit_vector)

    (
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        bdecay_tensor_arm,
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        bg_tensor_arm,
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        gen_tensor_arm,
        combined_tensor_delta_E,
        combined_tensor_annihilation_angle,
        combined_tensor_arm,
    ) = create_tensors(
        bdecay_list_delta_E,
        bdecay_list_annihilation_angle,
        bdecay_list_arm,
        bg_list_delta_E,
        bg_list_annihilation_angle,
        bg_list_arm,
        gen_list_delta_E,
        gen_list_annihilation_angle,
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
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        bdecay_tensor_arm,
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        bg_tensor_arm,
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        gen_tensor_arm,
        combined_tensor_delta_E,
        combined_tensor_annihilation_angle,
        combined_tensor_arm,
    )

    # Use the actual training class balance as the Bayesian prior rather than
    # hard-coded numbers. Count only finite entries so that NaN-filled rows
    # (events dropped by event_data_processing) do not inflate either class.
    n_beta_decay = int(torch.isfinite(bdecay_tensor_delta_E).sum().item())
    n_bg = int(torch.isfinite(bg_tensor_delta_E).sum().item())

    #n_beta_decay = 20000
    #n_bg = 50000

    predict(
        ground_truths,
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
        n_beta_decay,
        n_bg,
    )
