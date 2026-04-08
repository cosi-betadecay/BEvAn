import omegaconf
import ROOT as M
import torch
from tqdm import tqdm

from physics.annihilation_detection import R
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.matrix_calculations import annihilation_angle, arm, build_density_matrix, delta_E
from utils.reader_extraction import get_reader


def test_annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    ground_truths = []

    # Lists beta decay
    bdecay_list_delta_E = []
    bdecay_list_annihilation_angle = []
    bdecay_list_arm = []
    # Lists bg
    bg_list_delta_E = []
    bg_list_annihilation_angle = []
    bg_list_arm = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        gt = ground_truth_bdecay(event, ref_energy)
        ground_truths.append(gt)

        energies, positions = event_data_processing(event, cfg)
        if energies.numel() == 0 or positions.numel() == 0:
            bdecay_list_delta_E.append(0.0)
            bdecay_list_annihilation_angle.append(0.0)
            bdecay_list_arm.append(0.0)
            bg_list_delta_E.append(0.0)
            bg_list_annihilation_angle.append(0.0)
            bg_list_arm.append(0.0)
            continue

        _delta_E = delta_E(energies)
        _annihilation_angle = annihilation_angle(positions)
        _arm = arm(energies, positions, cfg)

        # Asserting that the different functions only return a tensor with 1 element
        assert _delta_E.numel() == 1
        assert _annihilation_angle.numel() == 1
        assert _arm.numel() == 1

        if gt:
            bdecay_list_delta_E.append(_delta_E)
            bdecay_list_annihilation_angle.append(_annihilation_angle)
            bdecay_list_arm.append(_arm)
        else:
            bg_list_delta_E.append(_delta_E)
            bg_list_annihilation_angle.append(_annihilation_angle)
            bg_list_arm.append(_arm)

    # Converting beta decay lists to tensors
    bdecay_tensor_delta_E = torch.tensor(bdecay_list_delta_E, dtype=torch.float32)
    bdecay_tensor_annihilation_angle = torch.tensor(bdecay_list_annihilation_angle, dtype=torch.float32)
    bdecay_tensor_arm = torch.tensor(bdecay_list_arm, dtype=torch.float32)
    # Converting bg lists to tensors
    bg_tensor_delta_E = torch.tensor(bg_list_delta_E, dtype=torch.float32)
    bg_tensor_annihilation_angle = torch.tensor(bg_list_annihilation_angle, dtype=torch.float32)
    bg_tensor_arm = torch.tensor(bg_list_arm, dtype=torch.float32)

    ground_truths = torch.tensor(ground_truths, dtype=torch.bool)

    # Asserting shape at tensor level
    assert bdecay_tensor_delta_E.shape == ground_truths.shape
    assert bdecay_tensor_annihilation_angle.shape == ground_truths.shape
    assert bdecay_tensor_arm.shape == ground_truths.shape
    assert bg_tensor_delta_E.shape == ground_truths.shape
    assert bg_tensor_annihilation_angle.shape == ground_truths.shape
    assert bg_tensor_arm.shape == ground_truths.shape

    # Building density matrices for beta decay
    bdecay_matrix_deltaE_vs_annihilation_angle = build_density_matrix(
        bdecay_tensor_delta_E, bdecay_tensor_annihilation_angle
    )
    bdecay_matrix_deltaE_vs_arm = build_density_matrix(bdecay_tensor_delta_E, bdecay_tensor_arm)
    # Building density matrices for background
    bg_matrix_deltaE_vs_annihilation_angle = build_density_matrix(
        bg_tensor_delta_E, bg_tensor_annihilation_angle
    )
    bg_matrix_deltaE_vs_arm = build_density_matrix(bg_tensor_delta_E, bg_tensor_arm)

    ratio = R(
        bdecay_matrix_deltaE_vs_annihilation_angle,
        bg_matrix_deltaE_vs_annihilation_angle,
        bdecay_matrix_deltaE_vs_arm,
        bg_matrix_deltaE_vs_arm,
    )
