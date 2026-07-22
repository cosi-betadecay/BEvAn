import torch
import wandb

from modeling.bayesian_classifier import BayesianClassifier
from modeling.metrics import metrics
from utils.wandb_logging import log_confusion_matrix


def log_R(terms: list[tuple[torch.Tensor, torch.Tensor]], eps: float = 1e-8) -> torch.Tensor:
    """Per-event log-likelihood ratio log[P(D|β)/P(D|bg)] from the term densities.

    Sums the log-density differences across ``terms`` (each a ``(p_beta, p_bg)``
    pair). ``eps`` guards against ``log(0)``. Stays in log space — the decision
    rule compares here directly, so the ratio is never exponentiated and cannot
    overflow no matter how many terms contribute.

    Args:
        terms: Per-term ``(p_beta, p_bg)`` density tensors, all event-aligned.
        eps: Additive floor inside the logs.

    Returns:
        The log-likelihood ratio per event, same shape as a term tensor.
    """
    log_r = torch.zeros_like(terms[0][0])
    for p_beta, p_bg in terms:
        log_r = log_r + torch.log(p_beta + eps) - torch.log(p_bg + eps)
    return log_r


def confusion_counts(
    ground_truths: torch.Tensor,
    terms: list[tuple[torch.Tensor, torch.Tensor]],
    n_beta_decay: int,
    n_bg: int,
) -> dict[str, int]:
    """Classify events via the Bayesian posterior and tally a confusion matrix.

    Builds the log-likelihood ratio from ``terms``, applies the class prior from
    ``n_beta_decay``/``n_bg``, and compares predictions against ``ground_truths``.
    Events with a NaN ratio are excluded.

    Args:
        ground_truths: Per-event boolean truth labels (β⁺ = True).
        terms: Per-term ``(p_beta, p_bg)`` densities (empty -> log-ratio of 0).
        n_beta_decay: β⁺ event count for the class prior.
        n_bg: Background event count for the class prior.

    Returns:
        Dict with ``tp``, ``fp``, ``fn``, ``tn``, and ``excluded`` counts.
    """
    ground_truths = torch.as_tensor(ground_truths, dtype=torch.bool)
    log_r = log_R(terms) if terms else torch.zeros(ground_truths.shape[0])

    predictions = BayesianClassifier(log_r, n_beta_decay, n_bg).decision_rule()
    valid = ~torch.isnan(log_r)
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


def log_confusion(counts: dict[str, int], split_name: str = "eval", log_image: bool = True) -> None:
    """Log precision/recall/FPR/F1 for ``counts`` to W&B.

    No-op when no W&B run is active. With ``log_image`` a confusion-matrix
    figure is logged too.

    Args:
        counts: Confusion counts (``tp``/``fp``/``fn``/``tn``/``excluded``).
        split_name: Metric namespace prefix (e.g. ``"eval"`` or ``"eval/n2"``).
        log_image: Whether to attach a confusion-matrix heatmap to the W&B log.
    """
    if wandb.run is None:
        return

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    scores = metrics(counts)
    precision, recall, fpr, f1_score = (
        scores["precision"],
        scores["recall"],
        scores["fpr"],
        scores["f1_score"],
    )

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
        }
    )
    if log_image:
        log_confusion_matrix(counts, split_name)
