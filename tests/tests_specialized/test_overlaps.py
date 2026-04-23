import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from modeling.bayesian_annihilation import BayesianAnnihiliationModel


def _finite_pair(beta_tensor, bg_tensor):
    a = beta_tensor[torch.isfinite(beta_tensor)].cpu().numpy().astype("float64")
    b = bg_tensor[torch.isfinite(bg_tensor)].cpu().numpy().astype("float64")
    return a, b


def _common_edges(a, b, n_bins, spacing):
    if spacing == "log":
        positive = np.concatenate([a[a > 0], b[b > 0]])
        if positive.size == 0:
            raise ValueError("no positive values for log spacing")
        floor = max(float(positive.min()), 1e-3)
        ceil = float(max(a.max(), b.max()))
        if floor >= ceil:
            ceil = floor + 1e-6
        return np.geomspace(floor, ceil, n_bins + 1)
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if lo == hi:
        hi = lo + 1e-6
    return np.linspace(lo, hi, n_bins + 1)


def _density(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype="float64")
    return counts.astype("float64") / total / widths


def _bhattacharyya(p, q, edges):
    widths = np.diff(edges)
    return float(np.sum(np.sqrt(p * q) * widths))


def _total_variation(p, q, edges):
    widths = np.diff(edges)
    return 0.5 * float(np.sum(np.abs(p - q) * widths))


def _f1(labels, predictions):
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _plot_overlap(name, beta_values, bg_values, edges, log_x):
    p_beta = _density(beta_values, edges)
    p_bg = _density(bg_values, edges)
    bc = _bhattacharyya(p_beta, p_bg, edges)
    tv = _total_variation(p_beta, p_bg, edges)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stairs(p_beta, edges, label=f"true β (n={beta_values.size})", alpha=0.6, fill=True)
    ax.stairs(p_bg, edges, label=f"true bg (n={bg_values.size})", alpha=0.6, fill=True)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"{name} class-conditional overlap (BC={bc:.3f}, TV={tv:.3f})")
    ax.legend()
    return fig, bc, tv


def test_arm_class_overlap(real_tensors, wandb_run):
    """ARM should land closer to 0 for true β than for bg events. If the
    distributions overlap heavily, the Compton reconstruction is too noisy to
    add discriminative signal beyond ΔE."""
    beta, bg = _finite_pair(real_tensors["bdecay_arm"], real_tensors["bg_arm"])
    edges = _common_edges(beta, bg, n_bins=80, spacing="log")
    fig, bc, tv = _plot_overlap("ARM", beta, bg, edges, log_x=True)

    wandb_run.log(
        {
            "overlap/arm/bhattacharyya": bc,
            "overlap/arm/total_variation": tv,
            "overlap/arm/n_beta": int(beta.size),
            "overlap/arm/n_bg": int(bg.size),
            "overlap/arm/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert tv > 0.05, (
        f"ARM total-variation distance between β and bg is only {tv:.3f} — "
        "the two class-conditional distributions are nearly identical, so the "
        "ARM reconstruction caps achievable F1 regardless of binning or smoothing."
    )


def test_annihilation_angle_class_overlap(real_tensors, wandb_run):
    """Annihilation cosine should differ between back-to-back β-decay
    topologies and unrelated background topologies. Heavy overlap means the
    angle reconstruction itself is too noisy to separate classes."""
    beta, bg = _finite_pair(real_tensors["bdecay_angle"], real_tensors["bg_angle"])
    edges = _common_edges(beta, bg, n_bins=80, spacing="linear")
    fig, bc, tv = _plot_overlap("annihilation cos(θ)", beta, bg, edges, log_x=False)

    wandb_run.log(
        {
            "overlap/angle/bhattacharyya": bc,
            "overlap/angle/total_variation": tv,
            "overlap/angle/n_beta": int(beta.size),
            "overlap/angle/n_bg": int(bg.size),
            "overlap/angle/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert tv > 0.05, (
        f"Annihilation-angle total-variation distance between β and bg is only "
        f"{tv:.3f} — the two distributions are nearly identical; the angle "
        "reconstruction caps achievable F1."
    )


def test_deltaE_class_overlap(real_tensors, wandb_run):
    """Control: ΔE is the primary discriminator. If even ΔE shows weak
    separation, the signal is not in the data at all — not a model issue."""
    beta, bg = _finite_pair(real_tensors["bdecay_dE"], real_tensors["bg_dE"])
    edges = _common_edges(beta, bg, n_bins=80, spacing="log")
    fig, bc, tv = _plot_overlap("ΔE", beta, bg, edges, log_x=True)

    wandb_run.log(
        {
            "overlap/deltaE/bhattacharyya": bc,
            "overlap/deltaE/total_variation": tv,
            "overlap/deltaE/n_beta": int(beta.size),
            "overlap/deltaE/n_bg": int(bg.size),
            "overlap/deltaE/hist": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert tv > 0.1, (
        f"ΔE total-variation distance between β and bg is only {tv:.3f} — too "
        "low for the primary discriminator; data has no usable signal."
    )


def test_deltaE_only_f1_vs_full(real_posterior_scores, wandb_run):
    """Strip the angle and ARM conditional terms from R and compute F1 from
    the ΔE marginal alone. If F1_full ≈ F1_dE_only, ARM and angle are not
    contributing discriminative signal beyond ΔE — the eval-F1 ceiling lives
    in the feature reconstruction, not the model."""
    n_bins = real_posterior_scores["n_bins"]
    valid = real_posterior_scores["valid"]
    labels = real_posterior_scores["labels"][valid]
    n_beta_decay = real_posterior_scores["n_beta_decay"]
    n_bg = real_posterior_scores["n_bg"]

    full_predictions = real_posterior_scores["model"].inference().detach().cpu().numpy().astype(bool)[valid]
    f1_full = _f1(labels, full_predictions)

    # ΔE-only ratio: R with the angle and ARM conditional terms set to 1
    # (so log(1)=0 contributes nothing to the log-likelihood ratio). Mirrors
    # the eps-floor convention used by R() in modeling.calculate_probablities.
    eps = 1e-8
    p_beta_deltaE = real_posterior_scores["p_beta_deltaE"]
    p_bg_deltaE = real_posterior_scores["p_bg_deltaE"]
    ratio_dE_only = (p_beta_deltaE + eps) / (p_bg_deltaE + eps)
    dE_predictions = (
        BayesianAnnihiliationModel(ratio_dE_only, n_beta_decay, n_bg)
        .inference()
        .detach()
        .cpu()
        .numpy()
        .astype(bool)[valid]
    )
    f1_dE_only = _f1(labels, dE_predictions)
    f1_lift = f1_full - f1_dE_only

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["ΔE only", "ΔE + angle + ARM"],
        [f1_dE_only, f1_full],
        color=["tab:gray", "tab:blue"],
    )
    for bar, v in zip(bars, [f1_dE_only, f1_full]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title(f"F1 attribution to ARM + angle (n_bins={n_bins})")

    wandb_run.log(
        {
            f"feature_ablation/n_bins={n_bins}/f1_full": f1_full,
            f"feature_ablation/n_bins={n_bins}/f1_dE_only": f1_dE_only,
            f"feature_ablation/n_bins={n_bins}/f1_lift_from_arm_angle": f1_lift,
            f"feature_ablation/n_bins={n_bins}/bars": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert f1_full > f1_dE_only + 0.02, (
        f"Full-feature F1 ({f1_full:.3f}) gives only +{f1_lift:.3f} over "
        f"ΔE-only F1 ({f1_dE_only:.3f}) at n_bins={n_bins} — ARM and "
        "annihilation angle are not adding discriminative signal beyond ΔE; "
        "reconstruction quality is the bottleneck, not the model."
    )
