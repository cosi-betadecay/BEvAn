import matplotlib.pyplot as plt
import torch
from utils.plots import plot_confusion_matrix

import wandb
from modeling.bayesian_annihilation import BayesianAnnihiliationModel


def R(terms, eps: float = 1e-8) -> torch.Tensor:
    """Per-event likelihood ratio R = P(D|β) / P(D|bg).

    ``terms`` is a list of ``(p_beta, p_bg)`` density lookups, one per feature
    factor of the likelihood ratio. R is their product, computed in log space:

        R = prod_i  p_beta_i / p_bg_i

    The caller decides which factors to include (see :class:`Evaluator`), so this
    stays agnostic to the feature set — ΔE, angle, ARM and the LOR score all
    enter as just another ``(p_beta, p_bg)`` pair.
    """
    log_r = torch.zeros_like(terms[0][0])
    for p_beta, p_bg in terms:
        log_r = log_r + torch.log(p_beta + eps) - torch.log(p_bg + eps)
    return torch.exp(log_r)


def predict(ground_truths, terms, n_beta_decay: int, n_bg: int, split_name: str = "eval"):
    """Classify events from the likelihood-ratio ``terms``, apply the class
    prior, and log the confusion matrix.

    ``n_beta_decay`` / ``n_bg`` are the training class counts used as the
    P(β)/P(bg) prior. ``split_name`` prefixes the W&B metric keys.
    """
    ratio = R(terms)
    predictions = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg).inference()
    ground_truths = torch.as_tensor(ground_truths, dtype=torch.bool)

    # Events with undefined evidence (NaN ratio, from empty/invalid subsets) have
    # no real prediction — exclude them so they don't show up as phantom TN/FN.
    # An infinite (non-NaN) ratio is overwhelmingly strong evidence, so it stays.
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
