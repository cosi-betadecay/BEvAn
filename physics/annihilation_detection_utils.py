import matplotlib.pyplot as plt
import ROOT as M
import torch
import wandb
from tqdm import tqdm

from physics.bayesian_annihilation import BayesianAnnihiliationModel
from physics.event_processing import event_data_processing, strip_padding_from_event
from physics.ground_truths import ground_truth_bdecay
from physics.matrix_calculations import (
    annihilation_angle,
    arm,
    build_density_matrix,
    delta_E,
    lookup_density_values,
)
from utils.plots import plot_confusion_matrix


def compute_event_features(cfg, ref_energy, reader):
    ground_truths = []

    # Lists beta decay
    bdecay_list_delta_E = []
    bdecay_list_annihilation_angle = []
    bdecay_list_arm = []
    # Lists bg
    bg_list_delta_E = []
    bg_list_annihilation_angle = []
    bg_list_arm = []
    # Lists general
    gen_list_delta_E = []
    gen_list_annihilation_angle = []
    gen_list_arm = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)
        n_hits = event.GetNHTs()
        event_id = event.GetID()

        gt = ground_truth_bdecay(event, ref_energy)
        ground_truths.append(gt)

        energies, positions = event_data_processing(event)
        energies, positions = strip_padding_from_event(energies, positions)
        if energies is None or positions is None:
            gen_list_delta_E.append(float("nan"))
            gen_list_annihilation_angle.append(float("nan"))
            gen_list_arm.append(float("nan"))
            if gt:
                bdecay_list_delta_E.append(float("nan"))
                bdecay_list_annihilation_angle.append(float("nan"))
                bdecay_list_arm.append(float("nan"))
            else:
                bg_list_delta_E.append(float("nan"))
                bg_list_annihilation_angle.append(float("nan"))
                bg_list_arm.append(float("nan"))
            continue

        _delta_E = delta_E(energies)
        _annihilation_angle = annihilation_angle(positions, n_hits)
        _arm = arm(energies, positions, cfg)

        gen_list_delta_E.append(_delta_E)
        gen_list_annihilation_angle.append(_annihilation_angle)
        gen_list_arm.append(_arm)

        if gt:
            bdecay_list_delta_E.append(_delta_E)
            bdecay_list_annihilation_angle.append(_annihilation_angle)
            bdecay_list_arm.append(_arm)
        else:
            bg_list_delta_E.append(_delta_E)
            bg_list_annihilation_angle.append(_annihilation_angle)
            bg_list_arm.append(_arm)
    return (
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
    )


def create_tensors(
    bdecay_list_delta_E,
    bdecay_list_annihilation_angle,
    bdecay_list_arm,
    bg_list_delta_E,
    bg_list_annihilation_angle,
    bg_list_arm,
    gen_list_delta_E,
    gen_list_annihilation_angle,
    gen_list_arm,
):
    bdecay_tensor_delta_E = torch.tensor(bdecay_list_delta_E, dtype=torch.float32)
    bdecay_tensor_annihilation_angle = torch.tensor(bdecay_list_annihilation_angle, dtype=torch.float32)
    bdecay_tensor_arm = torch.tensor(bdecay_list_arm, dtype=torch.float32)
    # Converting bg lists to tensors
    bg_tensor_delta_E = torch.tensor(bg_list_delta_E, dtype=torch.float32)
    bg_tensor_annihilation_angle = torch.tensor(bg_list_annihilation_angle, dtype=torch.float32)
    bg_tensor_arm = torch.tensor(bg_list_arm, dtype=torch.float32)
    # Converting general lists to tensors
    gen_tensor_delta_E = torch.tensor(gen_list_delta_E, dtype=torch.float32)
    gen_tensor_annihilation_angle = torch.tensor(gen_list_annihilation_angle, dtype=torch.float32)
    gen_tensor_arm = torch.tensor(gen_list_arm, dtype=torch.float32)

    combined_tensor_delta_E = torch.cat([bdecay_tensor_delta_E, bg_tensor_delta_E])
    combined_tensor_annihilation_angle = torch.cat(
        [bdecay_tensor_annihilation_angle, bg_tensor_annihilation_angle]
    )
    combined_tensor_arm = torch.cat([bdecay_tensor_arm, bg_tensor_arm])
    return (
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


def build_density_matrices(
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
):
    _, deltaE_angle_bins, angle_bins = build_density_matrix(
        combined_tensor_delta_E,
        combined_tensor_annihilation_angle,
    )
    _, deltaE_arm_bins, arm_bins = build_density_matrix(
        combined_tensor_delta_E,
        combined_tensor_arm,
    )

    bdecay_matrix_deltaE_vs_annihilation_angle, _, _ = build_density_matrix(
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
    )
    bdecay_matrix_deltaE_vs_arm, _, _ = build_density_matrix(
        bdecay_tensor_delta_E,
        bdecay_tensor_arm,
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
    )
    # Building density matrices for background
    bg_matrix_deltaE_vs_annihilation_angle, _, _ = build_density_matrix(
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
    )
    bg_matrix_deltaE_vs_arm, _, _ = build_density_matrix(
        bg_tensor_delta_E,
        bg_tensor_arm,
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
    )

    n_beta_deltaE_annihilation_angle = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        bdecay_matrix_deltaE_vs_annihilation_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    n_bg_deltaE_annihilation_angle = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        bg_matrix_deltaE_vs_annihilation_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    n_beta_deltaE_arm = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_arm,
        bdecay_matrix_deltaE_vs_arm,
        deltaE_arm_bins,
        arm_bins,
    )
    n_bg_deltaE_arm = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_arm,
        bg_matrix_deltaE_vs_arm,
        deltaE_arm_bins,
        arm_bins,
    )

    return (
        n_beta_deltaE_annihilation_angle,
        n_bg_deltaE_annihilation_angle,
        n_beta_deltaE_arm,
        n_bg_deltaE_arm,
    )


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


def predict(
    ground_truths,
    n_beta_deltaE_annihilation_angle,
    n_bg_deltaE_annihilation_angle,
    n_beta_deltaE_arm,
    n_bg_deltaE_arm,
):
    ratio = R(
        n_beta_deltaE_annihilation_angle,
        n_bg_deltaE_annihilation_angle,
        n_beta_deltaE_arm,
        n_bg_deltaE_arm,
    )
    predictions = BayesianAnnihiliationModel(ratio, 2360, 50572).inference()
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
