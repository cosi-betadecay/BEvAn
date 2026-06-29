import torch


def metrics(counts: dict) -> dict:
    """Derive precision/recall/FPR/F1 from raw confusion counts.

    The single definition of these rates;
    :func:`modeling.calculate_probabilities.log_confusion` reuses it (adding MCC)
    so the printed/logged and returned metrics can never drift apart.

    Args:
        counts: Confusion counts with ``tp``/``fp``/``fn``/``tn`` keys.

    Returns:
        Dict with ``precision``, ``recall``, ``fpr``, and ``f1_score``.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1_score": f1}


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Prior-free ROC AUC via the tie-aware Mann-Whitney rank statistic.

    Invariant to any monotonic transform of ``scores`` (so the log-ratio gives
    the same answer as the ratio). Equals ``P(score(pos) > score(neg))`` with ties
    counted as half.

    Args:
        scores: Per-event separating scores (e.g. the pooled log-ratio).
        labels: Per-event boolean truth labels (β⁺ = True).

    Returns:
        The ROC AUC, or NaN if either class is empty.
    """
    labels = labels.bool()
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    s_sorted = scores[order]
    _, inv, counts = torch.unique_consecutive(s_sorted, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts  # 0-based start index of each tie group
    avg_rank_group = starts.double() + (counts.double() + 1) / 2  # 1-based average rank
    ranks_sorted = avg_rank_group[inv]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted

    r_pos = ranks[labels].sum().item()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def f1_at_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    """F1 when predicting signal for every event with ``scores >= threshold``.

    Args:
        scores: Per-event separating scores.
        labels: Per-event boolean truth labels (β⁺ = True).
        threshold: Decision cut; ``scores >= threshold`` predicts signal.

    Returns:
        The F1 score, or NaN if there are no predicted-or-actual positives.
    """
    labels = labels.bool()
    pred = scores >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


def best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Best F1 over all cuts, with the cut that achieves it (tie-aware).

    Mirrors the threshold sweep behind ``best_f1`` but also returns the winning
    cut, so a decision rule can be deployed at it. Cuts fall only between distinct
    score values; the returned ``threshold`` is a score value such that predicting
    ``scores >= threshold`` reproduces the reported F1.

    Args:
        scores: Per-event separating scores.
        labels: Per-event boolean truth labels (β⁺ = True).

    Returns:
        ``(best_f1, threshold)``; ``(NaN, NaN)`` if there are no positives.
    """
    labels = labels.bool()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan"), float("nan")

    order = torch.argsort(scores, descending=True)
    s_sorted = scores[order]
    y_sorted = labels[order].double()
    tp_cum = torch.cumsum(y_sorted, 0)
    k = torch.arange(1, s_sorted.numel() + 1, dtype=torch.double)
    f1 = 2 * tp_cum / (k + n_pos)

    # Valid cuts: the last element, or where the next score differs (tie-aware).
    boundary = torch.ones(s_sorted.numel(), dtype=torch.bool)
    boundary[:-1] = s_sorted[:-1] != s_sorted[1:]
    f1_valid = torch.where(boundary, f1, torch.full_like(f1, -1.0))
    best_idx = int(torch.argmax(f1_valid).item())
    return float(f1[best_idx].item()), float(s_sorted[best_idx].item())
