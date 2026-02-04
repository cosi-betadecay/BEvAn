import numpy as np
import wandb


def as_np(vals) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


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
            "mean": mean,
            "median": median,
            "trimmed_mean_5_95": trimmed_mean,
            "std": std,
            "var": var,
            "coeff_var": coeff_var,
            "mad": mad,
            "mean_abs_dev": mean_abs_dev,
            "q01": q01,
            "q05": q05,
            "q10": q10,
            "q25": q25,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "iqr": iqr,
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
            "frac_ge_10pct": fraction_ge[0.1],
            "frac_ge_50pct": fraction_ge[0.5],
            "frac_ge_90pct": fraction_ge[0.9],
            "frac_ge_99pct": fraction_ge[0.99],
            "frac_le_10pct": fraction_le_10,
        }
    )
