import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


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


def as_np(vals) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.clip(arr, 0.0, 1.0)


def _log_image(key: str, fig: plt.Figure) -> None:
    # Log image and pin to summary so it shows up immediately on the dashboard.
    img = wandb.Image(fig)
    wandb.log({key: img})
    if wandb.run is not None:
        wandb.run.summary[key] = img
    plt.close(fig)


def histogram(likelihoods: list[float], key: str, bins: int = 50):
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, range=(0, 1), color="#4c72b0", edgecolor="black", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count")
    ax.set_title(f"{key} (hist)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - hist", fig)


def histogram_log(likelihoods: list[float], key: str, bins: int = 50):
    x = as_np(likelihoods)
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(x, bins=bins, range=(0, 1), color="#55a868", edgecolor="black", alpha=0.8, log=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{key} (hist, log10(count))")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - hist_logy", fig)


def cdf(likelihoods: list[float], key: str):
    x = np.sort(as_np(likelihoods))
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#c44e52")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("CDF")
    ax.set_title(f"{key} (CDF)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - cdf", fig)


def violin_plot(likelihoods: list[float], key: str):
    vals = np.asarray(likelihoods, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 0.0, 1.0)

    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot(vals, showmeans=True, showmedians=True)
    ax.set_ylabel("Likelihood")
    ax.set_xticks([])
    ax.set_title(f"{key} (violin)")

    _log_image(f"{key} - violin", fig)


def ecdf_slope(likelihoods: list[float], key: str, bins: int = 80):
    # Approximate ECDF slope as a density estimate (histogram normalized).
    x = as_np(likelihoods)
    if x.size == 0:
        return
    counts, edges = np.histogram(x, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(centers, counts, color="#8172b2")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Likelihood")
    ax.set_ylabel("ECDF slope proxy / density")
    ax.set_title(f"{key} (ECDF slope proxy / density)")
    ax.grid(True, linestyle="--", alpha=0.4)

    _log_image(f"{key} - ecdf_slope", fig)


# Calculations


def log_calculations(likelihoods: list[float], key: str):
    x = as_np(likelihoods)
    valid_count = x.size

    if valid_count == 0:
        return

    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x)
    var = np.var(x)

    percentiles = np.percentile(x, [1, 5, 10, 25, 75, 90, 95, 99])
    q01, q05, q10, q25, q75, q90, q95, q99 = percentiles
    iqr = q75 - q25

    mad = np.median(np.abs(x - median))
    mean_abs_dev = np.mean(np.abs(x - mean))
    coeff_var = std / mean if mean != 0 else np.nan

    centered = x - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    skewness = m3 / (m2**1.5) if m2 > 0 else np.nan
    kurtosis_excess = m4 / (m2 * m2) - 3 if m2 > 0 else np.nan

    trimmed_mask = (x >= q05) & (x <= q95)
    trimmed_mean = np.mean(x[trimmed_mask]) if np.any(trimmed_mask) else mean

    thresholds = [0.1, 0.5, 0.9, 0.99]
    fraction_ge = {thr: float(np.mean(x >= thr)) for thr in thresholds}
    fraction_le_10 = float(np.mean(x <= 0.1))

    wandb.log(
        {
            f"{key} - mean": mean,
            f"{key} - median": median,
            f"{key} - trimmed_mean_5_95": trimmed_mean,
            f"{key} - std": std,
            f"{key} - var": var,
            f"{key} - coeff_var": coeff_var,
            f"{key} - mad": mad,
            f"{key} - mean_abs_dev": mean_abs_dev,
            f"{key} - q01": q01,
            f"{key} - q05": q05,
            f"{key} - q10": q10,
            f"{key} - q25": q25,
            f"{key} - q75": q75,
            f"{key} - q90": q90,
            f"{key} - q95": q95,
            f"{key} - q99": q99,
            f"{key} - iqr": iqr,
            f"{key} - skewness": skewness,
            f"{key} - kurtosis_excess": kurtosis_excess,
            f"{key} - frac_ge_10pct": fraction_ge[0.1],
            f"{key} - frac_ge_50pct": fraction_ge[0.5],
            f"{key} - frac_ge_90pct": fraction_ge[0.9],
            f"{key} - frac_ge_99pct": fraction_ge[0.99],
            f"{key} - frac_le_10pct": fraction_le_10,
        }
    )
