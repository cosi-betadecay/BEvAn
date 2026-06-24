import math

import matplotlib.pyplot as plt
import torch

import wandb
from modeling.bayesian_annihilation import BayesianAnnihiliationModel
from utils.plots import plot_confusion_matrix


def R(terms: list[tuple[torch.Tensor, torch.Tensor]], eps: float = 1e-8) -> torch.Tensor:
    """Per-event likelihood ratio P(D|β)/P(D|bg) from the product of term densities.

    Sums the log-density differences across ``terms`` (each a ``(p_beta, p_bg)``
    pair) and exponentiates. ``eps`` guards against ``log(0)``.

    Args:
        terms: Per-term ``(p_beta, p_bg)`` density tensors, all event-aligned.
        eps: Additive floor inside the logs.

    Returns:
        The likelihood ratio per event, same shape as a term tensor.
    """
    log_r = torch.zeros_like(terms[0][0])
    for p_beta, p_bg in terms:
        log_r = log_r + torch.log(p_beta + eps) - torch.log(p_bg + eps)
    return torch.exp(log_r)


def confusion_counts(
    ground_truths: torch.Tensor,
    terms: list[tuple[torch.Tensor, torch.Tensor]],
    n_beta_decay: int,
    n_bg: int,
    log_threshold: float | None = None,
) -> dict:
    """Classify events via the Bayesian posterior and tally a confusion matrix.

    Builds the likelihood ratio from ``terms``, applies the class prior from
    ``n_beta_decay``/``n_bg``, and compares predictions against ``ground_truths``.
    Events with a NaN ratio or NaN truth are excluded.

    Args:
        ground_truths: Per-event boolean truth labels (β⁺ = True).
        terms: Per-term ``(p_beta, p_bg)`` densities (empty -> ratio of 1).
        n_beta_decay: β⁺ event count for the class prior.
        n_bg: Background event count for the class prior.
        log_threshold: Calibrated LLR cut for the decision; None uses the
            count-prior posterior rule (the default operating point).

    Returns:
        Dict with ``tp``, ``fp``, ``fn``, ``tn``, and ``excluded`` counts.
    """
    ground_truths = torch.as_tensor(ground_truths, dtype=torch.bool)
    ratio = R(terms) if terms else torch.ones(ground_truths.shape[0])

    predictions = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg).inference(log_threshold)
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


def log_confusion(counts: dict, split_name: str = "eval", log_image: bool = True) -> None:
    """Print precision/recall/FPR/F1/MCC for ``counts`` and log them to W&B.

    Console output always prints; W&B logging is skipped when no run is active.
    With ``log_image`` a confusion-matrix figure is logged too.

    Args:
        counts: Confusion counts (``tp``/``fp``/``fn``/``tn``/``excluded``).
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"eval/n2"``).
        log_image: Whether to attach a confusion-matrix heatmap to the W&B log.
    """
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

    # Console output always happens; wandb logging only when a run is active.
    if wandb.run is None:
        return

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
