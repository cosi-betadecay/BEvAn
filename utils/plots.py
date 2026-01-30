import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from matplotlib.colors import LogNorm

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


def energy_kernel_vs_total_energy(
    energy_kernels: torch.Tensor, total_energies: torch.Tensor, label: str
) -> None:
    """2D log-histogram plot of energy kernels vs total energies.

    Args:
        energy_kernels (torch.Tensor): A tensor of energy kernels.
        total_energies (torch.Tensor): A tensor of total energies.
    """
    kernels = energy_kernels.detach().cpu().numpy().reshape(-1)
    totals = total_energies.detach().cpu().numpy().reshape(-1)
    mask = np.isfinite(kernels) & np.isfinite(totals) & (totals > 0)
    kernels = kernels[mask]
    totals = totals[mask]

    num_kernel_bins = 20
    num_energy_bins = 20

    x_min = np.nanpercentile(kernels, 1)
    x_max = np.nanpercentile(kernels, 99)
    y_min = np.nanpercentile(totals[totals > 0], 1)
    y_max = np.nanpercentile(totals, 99)

    x_bins = np.linspace(x_min, x_max, num_kernel_bins + 1)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_energy_bins + 1)

    counts, x_edges, y_edges = np.histogram2d(
        kernels,
        totals,
        bins=[x_bins, y_bins],
        weights=kernels,
    )
    counts = np.ma.masked_where(counts <= 0, counts)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap="turbo",
        norm=LogNorm(),
        shading="auto",
    )
    ax.set_xlabel("Energy kernel")
    ax.set_ylabel("Total energy [keV]")
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("counts")

    fig.tight_layout()
    _log_image(label, fig)


def arm_kernel_vs_arm(arm_kernels: torch.Tensor, arms: torch.Tensor, label: str) -> None:
    """2D log-histogram plot of arm kernels vs arms.

    Args:
        arm_kernels (torch.Tensor): A tensor of arm kernels.
        arms (torch.Tensor): A tensor of arms.
    """
    kernels = arm_kernels.detach().cpu().numpy().reshape(-1)
    arm_vals = arms.detach().cpu().numpy().reshape(-1)

    valid = np.isfinite(kernels) & np.isfinite(arm_vals)
    kernels = kernels[valid]
    arm_vals = arm_vals[valid]

    num_kernel_bins = 20
    num_arm_bins = 20

    x_min = np.nanpercentile(kernels, 1)
    x_max = np.nanpercentile(kernels, 99)
    y_min = np.nanpercentile(arm_vals[arm_vals > 0], 1)
    y_max = np.nanpercentile(arm_vals, 99)

    x_bins = np.linspace(x_min, x_max, num_kernel_bins + 1)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_arm_bins + 1)

    counts, x_edges, y_edges = np.histogram2d(kernels, arm_vals, bins=[x_bins, y_bins])
    counts = np.ma.masked_where(counts <= 0, counts)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap="turbo",
        norm=LogNorm(),
        shading="auto",
    )
    ax.set_xlabel("ARM kernel")
    ax.set_ylabel("ARM [rad]")
    ax.set_xlim(0.0, 1.0)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("counts")

    fig.tight_layout()
    _log_image(label, fig)
