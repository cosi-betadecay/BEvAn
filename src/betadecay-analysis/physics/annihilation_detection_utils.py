"""β⁺-decay vs background classifier built on a Zoglauer-style Bayesian scheme.

This module adapts the Bayesian Compton Sequence Reconstruction (B-CSR)
methodology from Zoglauer's PhD §4.5.3. The original B-CSR classifies
*correct vs wrong interaction sequence* for a single γ-ray; here the same
likelihood-ratio framework is repurposed to classify *β⁺-decay vs background*
events.

Scope / current implementation
------------------------------
Sub-spaces used: (ΔE), (annihilation_angle | ΔE), (ARM | ΔE).
ΔE is factored out of the two 2D joints P(ΔE, y) via
``conditional_from_joint`` so it contributes to the likelihood ratio exactly
once, matching the ``Π_i p(m_i | C)`` independence assumption of eq. 4.26.

Intentionally omitted (future work, not part of the present implementation):
    * Compton-absorption probability a_C sub-space (§4.5.3 central interaction).
    * Photo-absorption probability a_P sub-space (final interaction).
    * Geometric distance factor d_E.
    * Detector-type dimension and interaction-multiplicity N.
    * Separate dθ-criterion sub-space for tracked electrons.

Likelihood-ratio form (Zoglauer eq. 4.26 adapted):
    R = [P(β)/P(bg)] · [P_β(ΔE)/P_bg(ΔE)]
                     · [P_β(angle|ΔE)/P_bg(angle|ΔE)]
                     · [P_β(ARM|ΔE)/P_bg(ARM|ΔE)]
    p(β | m) = R / (R + 1)
"""

import matplotlib.pyplot as plt
import ROOT as M
import torch
import wandb
from tqdm import tqdm
from utils.plots import plot_confusion_matrix

from physics.bayesian_annihilation import BayesianAnnihiliationModel
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
    lookup_density_values,
    lookup_density_values_1d,
)
from physics.matrix_calculations_factors import annihilation_angle, arm, delta_E


def compute_event_features(cfg, ref_energy, reader, reconstructed_unit_vector):
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

        energies, positions, sizes = event_data_processing(event)
        if energies is None or positions is None or sizes is None:
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

        _delta_E = delta_E(energies, sizes=sizes)
        _annihilation_angle = annihilation_angle(positions, n_hits=n_hits, sizes=sizes)
        _arm = arm(energies, positions, cfg, reconstructed_unit_vector, sizes=sizes)

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    """Build the Zoglauer-style likelihood factors for β vs bg classification.

    ΔE appears in both 2D joints (ΔE × angle, ΔE × ARM). To avoid
    double-counting it when the likelihood ratio multiplies the two
    sub-spaces, each 2D joint is factored as P(ΔE, y) = P(ΔE) · P(y | ΔE),
    and only one ΔE marginal is kept.

    Returns six per-event likelihood tensors ordered as
        (P_β(ΔE),   P_bg(ΔE),
         P_β(angle | ΔE), P_bg(angle | ΔE),
         P_β(ARM   | ΔE), P_bg(ARM   | ΔE))
    all of length ``len(gen_tensor_delta_E)``. Out-of-support events return
    0.0 (no evidence) — see ``lookup_density_values``.
    """
    # Shared bin edges derived from the combined (bdecay + bg) population so
    # that bdecay and bg matrices live on the same grid. ΔE and ARM are
    # residual-like quantities that peak near 0 with a long tail, so log
    # spacing concentrates resolution where the signal lives (Zoglauer B-CSR
    # §4.5.3 uses the same principle for energies and angle residuals). The
    # annihilation_angle axis is a bounded cosine and stays linear.
    # log_x_floor lowered from 0.1 to 1e-3 so well-reconstructed β⁺ events
    # (whose ΔE clusters well below 0.1 keV, inside the 3σ ≈ 2.87 keV
    # tolerance) are not all compressed into the first bin alongside merely
    # "lucky" background events.
    _, deltaE_angle_bins, angle_bins = build_density_matrix(
        combined_tensor_delta_E,
        combined_tensor_annihilation_angle,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=1e-3,
    )
    _, deltaE_arm_bins, arm_bins = build_density_matrix(
        combined_tensor_delta_E,
        combined_tensor_arm,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=1e-3,
        log_y_floor=1e-3,
        n_bins_x=10000,
        n_bins_y=10000,
    )

    # --- 2D joints per class (needed to derive the angle|ΔE and ARM|ΔE
    #     conditionals) ---
    # Laplace smoothing (α=1.0 per cell) is applied to the per-class joints
    # so that bins the minority class never populated still return a small
    # non-zero density at lookup time — otherwise a β⁺ test event landing in
    # an empty β⁺ bin contributes log(eps) ≈ −18 nats against itself, a
    # dominant silent source of false negatives.
    bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
        smoothing=1.0,
    )
    bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
        bdecay_tensor_delta_E,
        bdecay_tensor_arm,
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
        smoothing=1.0,
    )
    bg_joint_deltaE_angle, _, _ = build_density_matrix(
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
        smoothing=1.0,
    )
    bg_joint_deltaE_arm, _, _ = build_density_matrix(
        bg_tensor_delta_E,
        bg_tensor_arm,
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
        smoothing=1.0,
    )

    # --- ΔE marginals (1D). Use the same deltaE_angle_bins as the canonical
    #     ΔE grid; the arm-pair grid has the same support since it is derived
    #     from the same combined ΔE tensor, but may differ in the last bin
    #     due to floating-point ties. We build both 1D marginals independently
    #     so lookups use matching edges. ---
    bdecay_marginal_deltaE_angle_grid, _ = build_density_matrix_1d(
        bdecay_tensor_delta_E, x_bins=deltaE_angle_bins, smoothing=1.0
    )
    bg_marginal_deltaE_angle_grid, _ = build_density_matrix_1d(
        bg_tensor_delta_E, x_bins=deltaE_angle_bins, smoothing=1.0
    )

    # --- Conditionals P(y | ΔE) per class, by row-normalizing the joint. ---
    bdecay_cond_angle = conditional_from_joint(bdecay_joint_deltaE_angle, axis=0)
    bdecay_cond_arm = conditional_from_joint(bdecay_joint_deltaE_arm, axis=0)
    bg_cond_angle = conditional_from_joint(bg_joint_deltaE_angle, axis=0)
    bg_cond_arm = conditional_from_joint(bg_joint_deltaE_arm, axis=0)

    # --- Per-event lookups ---
    p_beta_deltaE = lookup_density_values_1d(
        gen_tensor_delta_E, bdecay_marginal_deltaE_angle_grid, deltaE_angle_bins
    )
    p_bg_deltaE = lookup_density_values_1d(
        gen_tensor_delta_E, bg_marginal_deltaE_angle_grid, deltaE_angle_bins
    )

    p_beta_angle_given_deltaE = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        bdecay_cond_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    p_bg_angle_given_deltaE = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_annihilation_angle,
        bg_cond_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    p_beta_arm_given_deltaE = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_arm,
        bdecay_cond_arm,
        deltaE_arm_bins,
        arm_bins,
    )
    p_bg_arm_given_deltaE = lookup_density_values(
        gen_tensor_delta_E,
        gen_tensor_arm,
        bg_cond_arm,
        deltaE_arm_bins,
        arm_bins,
    )

    return (
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
    )


def R(
    p_beta_deltaE: torch.Tensor,
    p_bg_deltaE: torch.Tensor,
    p_beta_angle_given_deltaE: torch.Tensor,
    p_bg_angle_given_deltaE: torch.Tensor,
    p_beta_arm_given_deltaE: torch.Tensor,
    p_bg_arm_given_deltaE: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Likelihood-ratio Π_i P(m_i | β) / P(m_i | bg) with ΔE counted once.

    Corresponds to Zoglauer eq. 4.26 without the ``p(C)/p(C̄)`` prior; the
    prior is applied downstream in :class:`BayesianAnnihiliationModel`.
    """
    log_r = (
        torch.log(p_beta_deltaE + eps)
        - torch.log(p_bg_deltaE + eps)
        + torch.log(p_beta_angle_given_deltaE + eps)
        - torch.log(p_bg_angle_given_deltaE + eps)
        + torch.log(p_beta_arm_given_deltaE + eps)
        - torch.log(p_bg_arm_given_deltaE + eps)
    )
    return torch.exp(log_r)


def predict(
    ground_truths,
    p_beta_deltaE,
    p_bg_deltaE,
    p_beta_angle_given_deltaE,
    p_bg_angle_given_deltaE,
    p_beta_arm_given_deltaE,
    p_bg_arm_given_deltaE,
    n_beta_decay: int,
    n_bg: int,
):
    """Classify events, apply the class prior, log diagnostics.

    ``n_beta_decay`` and ``n_bg`` are the *training* class counts — used as
    the ``p(β) / p(bg)`` prior in Bayes' theorem. They must be derived from
    the tensors passed to :func:`build_density_matrices`, not hard-coded.
    """
    ratio = R(
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
    )
    predictions = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg).inference()
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
