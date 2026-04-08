import omegaconf

from physics.annihilation_detection_utils import (
    build_density_matrices,
    compute_event_features,
    create_tensors,
    predict,
)
from utils.reader_extraction import get_reader


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
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
    ) = compute_event_features(cfg, ref_energy, reader)

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

    n_beta_deltaE_annihilation_angle, n_bg_deltaE_annihilation_angle, n_beta_deltaE_arm, n_bg_deltaE_arm = (
        build_density_matrices(
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
    )

    predict(
        ground_truths,
        n_beta_deltaE_annihilation_angle,
        n_bg_deltaE_annihilation_angle,
        n_beta_deltaE_arm,
        n_bg_deltaE_arm,
    )
