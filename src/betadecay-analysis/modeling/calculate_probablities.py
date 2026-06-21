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
    stays agnostic to the feature set — ΔE, angle and ARM all enter as just
    another ``(p_beta, p_bg)`` pair.
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
    invalid/empty subsets) are excluded, matching :func:`predict`.
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


def threshold_sweep(
    ground_truths,
    terms,
    n_beta_decay: int,
    n_bg: int,
    taus=(1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0),
) -> list[dict]:
    """Read-only FP/precision/recall/F1 tradeoff vs the decision threshold.

    The shipped decision is the MAP rule ``R >= n_bg/n_beta`` (``tau=1``). Here we
    multiply that threshold by ``tau``: ``predict beta iff R >= tau*(n_bg/n_beta)``.
    ``tau>1`` raises the bar, trading recall for precision (fewer FP). This does NOT
    change any logged metric — it only reports what each operating point would give,
    so a good n>=3 threshold can be chosen empirically before touching the model.
    """
    gt = torch.as_tensor(ground_truths, dtype=torch.bool)
    ratio = R(terms) if terms else torch.ones(gt.shape[0])
    valid = ~torch.isnan(ratio) & ~torch.isnan(gt.to(ratio.dtype))
    gt = gt[valid]
    ratio = ratio[valid]
    base = n_bg / max(n_beta_decay, 1)  # MAP threshold on R

    rows = []
    for tau in taus:
        pred = ratio >= tau * base
        tp = int((gt & pred).sum().item())
        fp = int((~gt & pred).sum().item())
        fn = int((gt & ~pred).sum().item())
        tn = int((~gt & ~pred).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {"tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "precision": precision, "recall": recall, "f1": f1}
        )
    return rows


def log_threshold_sweep(rows: list[dict], split_name: str = "eval"):
    """Pretty-print a :func:`threshold_sweep` table (diagnostic only, no W&B)."""
    print(f"{split_name} - threshold sweep (tau=1.0 is the shipped MAP rule):")
    print(f"{split_name} -   tau    TP    FP    FN     TN   Prec   Rec    F1")
    for r in rows:
        print(
            f"{split_name} - {r['tau']:5.2f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} "
            f"{r['tn']:6d} {r['precision']:.4f} {r['recall']:.4f} {r['f1']:.4f}"
        )


def log_confusion(counts: dict, split_name: str = "eval", log_image: bool = True):
    """Print + log a confusion-matrix summary from raw counts (the sum across
    buckets, or one bucket's counts for diagnostics)."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{split_name} - NaN excluded: {counts.get('excluded', 0)}")
    print(f"{split_name} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(
        f"{split_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}, F1 Score: {f1_score:.4f}"
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
