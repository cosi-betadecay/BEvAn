import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from utils.calculations import as_np


def plot_confusion_matrix(TP: int, FP: int, FN: int, TN: int):
    """Plot and save a confusion matrix heatmap.

    Args:
        TP (int): Number of true positive events.
        FP (int): Number of false positive events.
        FN (int): Number of false negative events.
        TN (int): Number of true negative events.
    """
    cm = np.array([[TP, FN], [FP, TN]])
    labels = ["Positive", "Negative"]

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix")
    return fig


def _log_image(key: str, fig: plt.Figure) -> None:
    """_summary_

    Args:
        key (str): _description_
        fig (plt.Figure): _description_
    """
    img = wandb.Image(fig)
    wandb.log({key: img})
    if wandb.run is not None:
        wandb.run.summary[key] = img
    plt.close(fig)


def histogram(likelihoods: list[float], key: str, bins: int = 50):
    """_summary_

    Args:
        likelihoods (list[float]): _description_
        key (str): _description_
        bins (int, optional): _description_. Defaults to 50.
    """
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, color="#4c72b0", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count")
    ax.set_title(f"{key} (hist)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image("hist", fig)


def histogram_log(likelihoods: list[float], key: str, bins: int = 50):
    """_summary_

    Args:
        likelihoods (list[float]): _description_
        key (str): _description_
        bins (int, optional): _description_. Defaults to 50.
    """
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, color="#55a868", edgecolor="black", alpha=0.8, log=True)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{key} (hist, log10(count))")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image("hist_logy", fig)


def cdf(likelihoods: list[float], key: str):
    """_summary_

    Args:
        likelihoods (list[float]): _description_
        key (str): _description_
    """
    x = np.sort(as_np(likelihoods))
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#c44e52")
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("CDF")
    ax.set_title(f"{key} (CDF)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image("cdf", fig)


def violin_plot(likelihoods: list[float], key: str):
    """_summary_

    Args:
        likelihoods (list[float]): _description_
        key (str): _description_
    """
    vals = np.asarray(likelihoods, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot(vals, showmeans=True, showmedians=True)
    ax.set_ylabel("Likelihood")
    ax.set_xticks([])
    ax.set_title(f"{key} (violin)")

    _log_image("violin", fig)


def ecdf_slope(likelihoods: list[float], key: str, bins: int = 80):
    """_summary_

    Args:
        likelihoods (list[float]): _description_
        key (str): _description_
        bins (int, optional): _description_. Defaults to 80.
    """
    x = as_np(likelihoods)
    if x.size == 0:
        return
    counts, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(centers, counts, color="#8172b2")
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("ECDF slope proxy / density")
    ax.set_title(f"{key} (ECDF slope proxy / density)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image("ecdf_slope", fig)


def plot_energy_vs_arm(prediction_scores, energy_sums, arms, label: str):
    energy = as_np(energy_sums).ravel()
    arm = as_np(arms).ravel()
    scores = as_np(prediction_scores).ravel()

    mask = np.isfinite(energy) & np.isfinite(arm) & np.isfinite(scores)

    energy = energy[mask]
    arm = arm[mask]
    scores = scores[mask]

    if energy.size == 0:
        return None

    weights = np.exp(scores)

    e_min, e_max = energy.min(), energy.max()
    e_pad = max(1.0, 0.02 * (e_max - e_min))
    energy_range = (e_min - e_pad, e_max + e_pad)

    # ARM (symmetric around 0)
    arm_max = np.max(np.abs(arm))
    arm_pad = max(1e-3, 0.02 * arm_max) if arm_max > 0 else 1.0
    arm_range = (-arm_max - arm_pad, arm_max + arm_pad)

    # --- Binning (explicit, physical) ---

    # Energy bins: fine around 511 keV
    n_energy_bins = 20
    energy_bins = np.linspace(
        energy_range[0],
        energy_range[1],
        n_energy_bins + 1,
    )

    # ARM bins: symmetric around 0
    n_arm_bins = 20
    arm_bins = np.linspace(
        arm_range[0],
        arm_range[1],
        n_arm_bins + 1,
    )

    H, xedges, yedges = np.histogram2d(
        energy,
        arm,
        bins=[energy_bins, arm_bins],
        weights=weights,
    )

    # H = gaussian_filter(H, sigma=1.0)

    # LogNorm requires strictly positive values
    if not np.any(H > 0):
        return None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        H.T,
        norm=mcolors.LogNorm(
            vmin=np.min(H[H > 0]),
            vmax=np.max(H),
        ),
        cmap="cividis",
        shading="auto",
    )

    # Reference lines (physics anchors)
    ax.axvline(511.0, color="tab:orange", linestyle="--", linewidth=1.5)
    ax.axhline(0.0, color="tab:blue", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Reconstructed Energy [keV]")
    ax.set_ylabel("ARM [rad]")
    ax.set_title(label)
    ax.set_xlim(energy_range)
    ax.set_ylim(arm_range)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Posterior mass")

    _log_image("energy_vs_arm", fig)
