"""
1) ROC curve (TPR vs FPR)
What: parameterize threshold from 0→1, plot TPR(t) vs FPR(t), report AUC.
Based on: the raw posterior P(β|D) (or equivalently R) for every event, paired with the ground-truth label.
Tells you: the discriminative power of your likelihood model alone. Prior has no effect on ROC. If ROC is bad, the features or density estimate are bad. If ROC is good, any prior-induced misbehavior is a thresholding choice, not a model failure.

2) Precision-Recall curve
What: P vs R across thresholds; report average precision (AP).
Based on: same scores + labels as ROC.
Tells you: for imbalanced problems PR is more honest than ROC. If your β fraction is, say, 30%, ROC can look good while PR reveals the classifier gets crushed at high recall. Physics audiences tend to prefer efficiency/purity versions of this (same math, different axes).

3) Score distribution per class
What: two overlaid histograms of P(β|D) (or log R) — one for true β events, one for true bg.
Based on: same scores + labels.
Tells you: where the separation happens. A bimodal picture with β peaking near 1 and bg near 0 = good. Heavy overlap near 0.5 = weak features. Spikes at exactly 0 and 1 = you're in the degenerate sparse-bin regime (events whose bin contained only their own class).

Use these libraries at least:
pytest
wandb
matplotlib
USE THE REAL SIMULATION FILES IN DATA!!! MEANING WE SHOULD USE THE ROOT LIBRARY TOO.
...
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

# --------------------------------------------------------------------------- #
# Metric helpers (NumPy implementations — no sklearn dependency)
# --------------------------------------------------------------------------- #


def _roc_curve(labels, scores):
    """Return (FPR, TPR) arrays, sorted by decreasing threshold."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.int64)

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    P = int((labels == 1).sum())
    N = int((labels == 0).sum())

    tpr = np.concatenate(([0.0], tps / max(P, 1)))
    fpr = np.concatenate(([0.0], fps / max(N, 1)))
    return fpr, tpr


def _auc(x, y):
    """Trapezoidal AUC for monotonically non-decreasing x."""
    return float(np.trapz(y, x))


def _precision_recall_curve(labels, scores):
    """Return (precision, recall) at every distinct threshold, recall-sorted."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.int64)

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = int((labels == 1).sum())

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(P, 1)

    # Append the (precision=1, recall=0) endpoint expected for a PR curve.
    precision = np.concatenate((precision, [1.0]))
    recall = np.concatenate((recall, [0.0]))
    return precision, recall


def _average_precision(labels, scores):
    """AP = Σ_n (R_n − R_{n−1}) · P_n over thresholds (sklearn convention)."""
    precision, recall = _precision_recall_curve(labels, scores)
    # Walk from high-recall to low-recall end and integrate.
    p = precision[:-1]  # drop the (1, 0) endpoint for the sum
    r = recall[:-1]
    dr = np.diff(np.concatenate(([0.0], r)))
    return float(np.sum(p * dr))


# --------------------------------------------------------------------------- #
# ROC Curve
# --------------------------------------------------------------------------- #


def test_roc_curve(real_posterior_scores, wandb_run):
    scores = real_posterior_scores["scores"]
    labels = real_posterior_scores["labels"]

    fpr, tpr = _roc_curve(labels, scores)
    auc = _auc(fpr, tpr)

    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert auc > 0.5, f"ROC AUC = {auc:.3f} — classifier is at or below chance"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — P(β | D) vs truth (Activation.sim)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    wandb_run.log({"roc/auc": auc, "roc/curve": wandb.Image(fig)})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Precision-Recall Curve
# --------------------------------------------------------------------------- #


def test_precision_recall_curve(real_posterior_scores, wandb_run):
    scores = real_posterior_scores["scores"]
    labels = real_posterior_scores["labels"]

    precision, recall = _precision_recall_curve(labels, scores)
    ap = _average_precision(labels, scores)
    prevalence = float(labels.mean())

    assert precision.shape == recall.shape
    assert precision[-1] == 1.0 and recall[-1] == 0.0
    assert 0.0 <= ap <= 1.0
    assert ap > prevalence, f"AP ({ap:.3f}) did not beat the prevalence baseline ({prevalence:.3f})"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"baseline (prevalence = {prevalence:.2f})",
    )
    ax.set_xlabel("Recall (efficiency)")
    ax.set_ylabel("Precision (purity)")
    ax.set_title("Precision-Recall — P(β | D) vs truth (Activation.sim)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")

    wandb_run.log(
        {
            "pr/average_precision": ap,
            "pr/prevalence": prevalence,
            "pr/curve": wandb.Image(fig),
        }
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Score Distribution per Class
# --------------------------------------------------------------------------- #


def test_score_distribution_per_class(real_posterior_scores, wandb_run):
    scores = real_posterior_scores["scores"]
    labels = real_posterior_scores["labels"]

    beta_scores = scores[labels == 1]
    bg_scores = scores[labels == 0]

    assert beta_scores.size > 0 and bg_scores.size > 0

    mean_beta = float(beta_scores.mean())
    mean_bg = float(bg_scores.mean())
    separation = mean_beta - mean_bg
    edge_frac = float(np.mean((scores == 0.0) | (scores == 1.0)))

    assert mean_beta > mean_bg, (
        f"True-β posterior mean ({mean_beta:.3f}) should exceed true-bg posterior mean ({mean_bg:.3f})"
    )

    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        bg_scores,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true bg (n={bg_scores.size}, μ={mean_bg:.2f})",
    )
    ax.hist(
        beta_scores,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true β (n={beta_scores.size}, μ={mean_beta:.2f})",
    )
    ax.set_xlabel("P(β | D)")
    ax.set_ylabel("density")
    ax.set_title("Score distribution per class (Activation.sim)")
    ax.legend(loc="upper center")

    wandb_run.log(
        {
            "score_dist/mean_beta": mean_beta,
            "score_dist/mean_bg": mean_bg,
            "score_dist/separation": separation,
            "score_dist/edge_fraction_0_or_1": edge_frac,
            "score_dist/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)
