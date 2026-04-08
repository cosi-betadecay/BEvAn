import matplotlib.pyplot as plt
import omegaconf
import ROOT as M
import torch
import wandb
from tqdm import tqdm

from physics.bayesian_annihilation import BayesianAnnihiliationModel
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.matrix_calculations import (
    build_density_matrix,
    delta_E_and_annihilation_angle,
    delta_E_and_compton_cone,
)
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def R(
    n_beta_deltaE_annihilation_angle: torch.Tensor,
    n_bg_deltaE_annihilation_angle: torch.Tensor,
    n_beta_deltaE_arm: torch.Tensor,
    n_bg_deltaE_arm: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    log_r = (
        torch.log(n_beta_deltaE_annihilation_angle + eps)
        - torch.log(n_bg_deltaE_annihilation_angle + eps)
        + torch.log(n_beta_deltaE_arm + eps)
        - torch.log(n_bg_deltaE_arm + eps)
    )
    return torch.exp(log_r)


def annihilation_extractor(
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

        energies, positions = event_data_processing(event, cfg)
        gt = ground_truth_bdecay(event, ref_energy)
        ground_truths.append(gt)

        _delta_E, _annihilation_angle = delta_E_and_annihilation_angle(energies, positions)
        _, _arm = delta_E_and_compton_cone(energies, positions, cfg)

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
    predictions = BayesianAnnihiliationModel(ratio, 3000, 50000).inference()
    ground_truths = torch.tensor(ground_truths, dtype=torch.bool)

    tp = torch.sum(ground_truths & predictions).item()
    fp = torch.sum(~ground_truths & predictions).item()
    fn = torch.sum(ground_truths & ~predictions).item()
    tn = torch.sum(~ground_truths & ~predictions).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fig = plot_confusion_matrix(tp, fp, fn, tn)
    wandb.log(
        {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "f1_score": f1_score,
            "confusion_matrix": wandb.Image(fig),
        }
    )
    plt.close(fig)
