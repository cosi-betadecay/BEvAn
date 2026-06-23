import math

import matplotlib.pyplot as plt
import torch
from utils.plots import plot_confusion_matrix

import wandb
from modeling.bayesian_annihilation import BayesianAnnihiliationModel


def R(terms, eps: float = 1e-8) -> torch.Tensor:
    """Per-event likelihood ratio R = P(D|β) / P(D|bg).

    ``terms`` is a list of ``(p_beta, p_bg)`` density lookups, one per feature
    factor. R is the naive-Bayes product over the factors, computed in log space:

        log R = sum_i  (log p_beta_i - log p_bg_i)
    """
    log_r = torch.zeros_like(terms[0][0])
    for p_beta, p_bg in terms:
        log_r = log_r + torch.log(p_beta + eps) - torch.log(p_bg + eps)
    return torch.exp(log_r)


def confusion_counts(ground_truths, terms, n_beta_decay: int, n_bg: int) -> dict:
    """Classify and return raw confusion counts (no logging), for a single
    bucket. ``terms`` may be empty (prior-only bucket), in which case the
    likelihood ratio is 1 everywhere and the per-bucket class prior decides.

    Returns ``{"tp","fp","fn","tn","excluded"}``. Events with a NaN ratio (e.g.
    invalid/empty subsets) are excluded.
    """
    ground_truths = torch.as_tensor(ground_truths, dtype=torch.bool)
    if terms:
        ratio = R(terms)
    else:
        ratio = torch.ones(ground_truths.shape[0])

    predictions = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg).inference()
    valid = ~torch.isnan(ratio) & ~torch.isnan(ground_truths.to(ratio.dtype))
    excluded = int((~valid).sum().item())
    gt = ground_truths[valid]
    pred = predictions[valid]
    return {
        "tp": int(torch.sum(gt & pred).item()),
        "fp": int(torch.sum(~gt & pred).item()),
        "fn": int(torch.sum(gt & ~pred).item()),
        "tn": int(torch.sum(~gt & ~pred).item()),
        "excluded": excluded,
    }


def log_confusion(counts: dict, split_name: str = "eval", log_image: bool = True):
    """Print + log a confusion-matrix summary from raw counts (the sum across
    buckets, or one bucket's counts for diagnostics)."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    print(f"{split_name} - NaN excluded: {counts.get('excluded', 0)}")
    print(f"{split_name} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(
        f"{split_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}, "
        f"F1 Score: {f1_score:.4f}, MCC: {mcc:.4f}"
    )

    log = {
        f"{split_name}/TP": tp,
        f"{split_name}/FP": fp,
        f"{split_name}/FN": fn,
        f"{split_name}/TN": tn,
        f"{split_name}/precision": precision,
        f"{split_name}/recall": recall,
        f"{split_name}/false_positive_rate": fpr,
        f"{split_name}/f1_score": f1_score,
    }
    if log_image:
        fig = plot_confusion_matrix(tp, fp, fn, tn)
        log[f"{split_name}/confusion_matrix"] = wandb.Image(fig)
        plt.close(fig)
    wandb.log(log)
