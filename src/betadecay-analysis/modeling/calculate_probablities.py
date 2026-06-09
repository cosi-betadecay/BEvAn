import matplotlib.pyplot as plt
import torch
import wandb
from utils.plots import plot_confusion_matrix

from modeling.bayesian_annihilation import BayesianAnnihiliationModel
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d


def get_probabilities(
    bdecay_marginal_deltaE_angle_grid,
    bg_marginal_deltaE_angle_grid,
    bdecay_cond_angle,
    bdecay_cond_arm,
    bg_cond_angle,
    bg_cond_arm,
    gen_tensor_delta_E,
    gen_tensor_annihilation_angle,
    gen_tensor_arm,
    deltaE_angle_bins,
    angle_bins,
    deltaE_arm_bins,
    arm_bins,
):
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
    p_beta_arm_given_deltaE: torch.Tensor | None = None,
    p_bg_arm_given_deltaE: torch.Tensor | None = None,
    eps: float = 1e-8,
    double_count: bool = False,
) -> torch.Tensor:
    """Compute the per-event likelihood ratio R = P(D|β) / P(D|bg).

    When ``double_count`` is False (default), this uses the conditional
    factorization that avoids double-counting ΔE:

        R = [P(ΔE|β)/P(ΔE|bg)]
            · [P(angle|ΔE,β)/P(angle|ΔE,bg)]
            · [P(ARM|ΔE,β)/P(ARM|ΔE,bg)]

    and expects the six conditional/marginal lookups in the standard order.

    When ``double_count`` is True, the first four arguments are reinterpreted
    as the two 2D joints and the last two arguments are ignored:

        p_beta_deltaE          → P(ΔE, angle | β)
        p_bg_deltaE            → P(ΔE, angle | bg)
        p_beta_angle_given_dE  → P(ΔE, ARM   | β)
        p_bg_angle_given_dE    → P(ΔE, ARM   | bg)

    giving R = [P(ΔE,angle|β)/P(ΔE,angle|bg)] · [P(ΔE,ARM|β)/P(ΔE,ARM|bg)],
    which double-counts ΔE.
    """
    if double_count:
        log_r = (
            torch.log(p_beta_deltaE + eps)
            - torch.log(p_bg_deltaE + eps)
            + torch.log(p_beta_angle_given_deltaE + eps)
            - torch.log(p_bg_angle_given_deltaE + eps)
        )
        return torch.exp(log_r)

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
    split_name: str = "eval",
    double_count: bool = False,
):
    """Classify events, apply the class prior, log diagnostics.

    ``n_beta_decay`` and ``n_bg`` are the *training* class counts — used as
    the ``p(β) / p(bg)`` prior in Bayes' theorem. They must be derived from
    the tensors passed to :func:`build_density_matrices`, not hard-coded.

    ``split_name`` prefixes every W&B metric key (e.g. ``"train/f1_score"``
    vs ``"eval/f1_score"``) so calls on different splits in the same run do
    not overwrite each other.

    ``double_count`` selects the likelihood-ratio formula; see :func:`R`.
    In double-counting mode the last two arguments may be ``None``.
    """
    ratio = R(
        p_beta_deltaE,
        p_bg_deltaE,
        p_beta_angle_given_deltaE,
        p_bg_angle_given_deltaE,
        p_beta_arm_given_deltaE,
        p_bg_arm_given_deltaE,
        double_count=double_count,
    )
    predictions = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg).inference()
    ground_truths = torch.as_tensor(ground_truths, dtype=torch.bool)

    # Exclude events whose evidence is undefined before tallying. Empty/invalid
    # subsets propagate NaN into the feature tensors and hence into the
    # likelihood ratio; such events have no real prediction. Counting them as
    # TN/FN turns NaN-reclassification into phantom confusion-matrix entries
    # that contaminate the metric. Mask out NaN ratios (and any NaN ground-truth
    # label) so TP/FP/FN/TN reflect genuine predictions only. Note: a non-NaN
    # but infinite ratio is overwhelmingly strong evidence, not undefined, so it
    # is intentionally kept.
    valid = ~torch.isnan(ratio) & ~torch.isnan(ground_truths.to(ratio.dtype))
    n_excluded = int((~valid).sum().item())
    ground_truths = ground_truths[valid]
    predictions = predictions[valid]
    print(f"{split_name} - NaN excluded: {n_excluded}")

    tp = torch.sum(ground_truths & predictions).item()
    fp = torch.sum(~ground_truths & predictions).item()
    fn = torch.sum(ground_truths & ~predictions).item()
    tn = torch.sum(~ground_truths & ~predictions).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{split_name} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(
        f"{split_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}, F1 Score: {f1_score:.4f}"
    )

    fig = plot_confusion_matrix(tp, fp, fn, tn)
    wandb.log(
        {
            f"{split_name}/TP": tp,
            f"{split_name}/FP": fp,
            f"{split_name}/FN": fn,
            f"{split_name}/TN": tn,
            f"{split_name}/precision": precision,
            f"{split_name}/recall": recall,
            f"{split_name}/false_positive_rate": fpr,
            f"{split_name}/f1_score": f1_score,
            f"{split_name}/confusion_matrix": wandb.Image(fig),
        }
    )
    plt.close(fig)
