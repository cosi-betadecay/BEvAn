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
    model = real_posterior_scores["model"]
    valid = real_posterior_scores["valid"]
    n_bins = real_posterior_scores["n_bins"]
    scores = model.beta_decay_given_data_probability().detach().cpu().numpy().astype("float64")[valid]
    labels = real_posterior_scores["labels"][valid]

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
    ax.set_title(f"ROC — P(β | D) vs truth (n_bins={n_bins})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    wandb_run.log(
        {
            f"roc/n_bins={n_bins}/auc": auc,
            f"roc/n_bins={n_bins}/curve": wandb.Image(fig),
        }
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Precision-Recall Curve
# --------------------------------------------------------------------------- #


def test_precision_recall_curve(real_posterior_scores, wandb_run):
    model = real_posterior_scores["model"]
    valid = real_posterior_scores["valid"]
    n_bins = real_posterior_scores["n_bins"]
    scores = model.beta_decay_given_data_probability().detach().cpu().numpy().astype("float64")[valid]
    labels = real_posterior_scores["labels"][valid]

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
    ax.set_title(f"Precision-Recall — P(β | D) vs truth (n_bins={n_bins})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")

    wandb_run.log(
        {
            f"pr/n_bins={n_bins}/average_precision": ap,
            f"pr/n_bins={n_bins}/prevalence": prevalence,
            f"pr/n_bins={n_bins}/curve": wandb.Image(fig),
        }
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Score Distribution per Class
# --------------------------------------------------------------------------- #


def test_score_distribution_per_class(real_posterior_scores, wandb_run):
    model = real_posterior_scores["model"]
    valid = real_posterior_scores["valid"]
    n_bins = real_posterior_scores["n_bins"]
    scores = model.beta_decay_given_data_probability().detach().cpu().numpy().astype("float64")[valid]
    labels = real_posterior_scores["labels"][valid]

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
    ax.set_title(f"Score distribution per class — P(β | D) (n_bins={n_bins})")
    ax.legend(loc="upper center")

    wandb_run.log(
        {
            f"score_dist/n_bins={n_bins}/mean_beta": mean_beta,
            f"score_dist/n_bins={n_bins}/mean_bg": mean_bg,
            f"score_dist/n_bins={n_bins}/separation": separation,
            f"score_dist/n_bins={n_bins}/edge_fraction_0_or_1": edge_frac,
            f"score_dist/n_bins={n_bins}/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Score Distribution per Class — P(bg | D)
# --------------------------------------------------------------------------- #


def test_score_distribution_per_class_bg(real_posterior_scores, wandb_run):
    model = real_posterior_scores["model"]
    valid = real_posterior_scores["valid"]
    n_bins = real_posterior_scores["n_bins"]
    bg_scores_all = model.background_given_data_probability().detach().cpu().numpy().astype("float64")[valid]
    labels = real_posterior_scores["labels"][valid]

    beta_scores = bg_scores_all[labels == 1]
    bg_scores = bg_scores_all[labels == 0]

    assert beta_scores.size > 0 and bg_scores.size > 0

    mean_beta = float(beta_scores.mean())
    mean_bg = float(bg_scores.mean())
    separation = mean_bg - mean_beta
    edge_frac = float(np.mean((bg_scores_all == 0.0) | (bg_scores_all == 1.0)))

    # For P(bg | D) the direction of separation flips: true bg events should
    # have a higher P(bg | D) than true β events.
    assert mean_bg > mean_beta, (
        f"True-bg posterior mean ({mean_bg:.3f}) should exceed true-β posterior mean ({mean_beta:.3f})"
    )

    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        beta_scores,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true β (n={beta_scores.size}, μ={mean_beta:.2f})",
    )
    ax.hist(
        bg_scores,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"true bg (n={bg_scores.size}, μ={mean_bg:.2f})",
    )
    ax.set_xlabel("P(bg | D)")
    ax.set_ylabel("density")
    ax.set_title(f"Score distribution per class — P(bg | D) (n_bins={n_bins})")
    ax.legend(loc="upper center")

    wandb_run.log(
        {
            f"score_dist_bg/n_bins={n_bins}/mean_beta": mean_beta,
            f"score_dist_bg/n_bins={n_bins}/mean_bg": mean_bg,
            f"score_dist_bg/n_bins={n_bins}/separation": separation,
            f"score_dist_bg/n_bins={n_bins}/edge_fraction_0_or_1": edge_frac,
            f"score_dist_bg/n_bins={n_bins}/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)
