import csv
import math
from pathlib import Path

# Default output directory: ablations/tables (created on first write).
TABLES_DIR = Path(__file__).resolve().parent / "tables"

# The metric keys shared across the comparison CSVs (record keys from the pipeline).
METRICS = ("f1_score", "auc", "precision", "recall")


def mean(values: list[float]) -> float:
    """Mean of the finite entries (NaN if there are none)."""
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def save_comparison_csv(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float]]],
    save_dir: Path = TABLES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Write one comparison ablation's per-dataset metrics, with a trailing mean row.

    Columns are ``dataset`` then ``our_model_<m>`` / ``ablation_<m>`` for each metric
    in :data:`METRICS`. Every dataset gets a row; only the trailing mean excludes
    ``exclude`` (so the mean matches the averaged plot while the detail stays whole).

    Args:
        ablation_name: Ablation label and output filename stem.
        per_dataset: ``{dataset: {"our_model": record, "ablation": record}}``.
        save_dir: Directory to write the CSV into.
        exclude: Dataset names dropped from the mean row only (kept as detail rows).

    Returns:
        Path: The written ``<ablation_name>.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{ablation_name}.csv"
    fieldnames = ["dataset"] + [f"{side}_{m}" for m in METRICS for side in ("our_model", "ablation")]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, rec in per_dataset.items():
            row = {"dataset": dataset}
            for m in METRICS:
                row[f"our_model_{m}"] = rec["our_model"].get(m, float("nan"))
                row[f"ablation_{m}"] = rec["ablation"].get(m, float("nan"))
            writer.writerow(row)
        included = [r for k, r in per_dataset.items() if k not in exclude]
        mean_row = {"dataset": f"mean (excl. {', '.join(exclude)})" if exclude else "mean"}
        for m in METRICS:
            mean_row[f"our_model_{m}"] = mean([r["our_model"].get(m, float("nan")) for r in included])
            mean_row[f"ablation_{m}"] = mean([r["ablation"].get(m, float("nan")) for r in included])
        writer.writerow(mean_row)
    return path


def save_factor_contributions_csv(
    contributions_by_dataset: dict[str, dict[str, dict[str, float]]],
    full_model_by_dataset: dict[str, dict[str, float]] | None = None,
    save_dir: Path = TABLES_DIR,
) -> Path:
    """Write per-dataset, per-factor F1/AUC in long format for pivoting.

    Columns: ``dataset``, ``factor``, ``f1``, ``auc``, and the full-model
    ``full_f1`` / ``full_auc`` reference for that dataset.

    Args:
        contributions_by_dataset: ``{dataset: {factor: {"f1": ..., "auc": ...}}}``.
        full_model_by_dataset: Optional ``{dataset: {"f1": ..., "auc": ...}}``.
        save_dir: Directory to write the CSV into.

    Returns:
        Path: The written ``factor_contributions.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "factor_contributions.csv"
    full_model_by_dataset = full_model_by_dataset or {}

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "factor", "f1", "auc", "full_f1", "full_auc"])
        writer.writeheader()
        for dataset, contrib in contributions_by_dataset.items():
            full = full_model_by_dataset.get(dataset, {})
            for factor, vals in contrib.items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "factor": factor,
                        "f1": vals.get("f1", float("nan")),
                        "auc": vals.get("auc", float("nan")),
                        "full_f1": full.get("f1", float("nan")),
                        "full_auc": full.get("auc", float("nan")),
                    }
                )
    return path


def save_gt_tolerance_csv(
    results_by_dataset: dict[str, dict[int, dict[str, float]]],
    save_dir: Path = TABLES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Write the label-window sweep in long format: one row per (dataset, n_std).

    Columns are ``dataset``, ``n_std``, then the :data:`METRICS`. Per-dataset rows
    keep every dataset; trailing ``mean`` rows (one per swept ``n_std``) average the
    non-``exclude`` datasets, matching the averaged plot.

    Args:
        results_by_dataset: ``{dataset: {n_std: metric_record}}``.
        save_dir: Directory to write the CSV into.
        exclude: Dataset names dropped from the mean rows only (kept as detail rows).

    Returns:
        Path: The written ``gt_tolerance.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "gt_tolerance.csv"
    n_std_values = sorted({n for per_n in results_by_dataset.values() for n in per_n})

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "n_std", *METRICS])
        writer.writeheader()
        for dataset, per_n in results_by_dataset.items():
            for n_std in sorted(per_n):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "n_std": n_std,
                        **{m: per_n[n_std].get(m, float("nan")) for m in METRICS},
                    }
                )
        label = f"mean (excl. {', '.join(exclude)})" if exclude else "mean"
        for n_std in n_std_values:
            included = [
                per_n[n_std] for k, per_n in results_by_dataset.items() if k not in exclude and n_std in per_n
            ]
            writer.writerow(
                {
                    "dataset": label,
                    "n_std": n_std,
                    **{m: mean([r.get(m, float("nan")) for r in included]) for m in METRICS},
                }
            )
    return path


def summary_row(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float]]],
    exclude: tuple[str, ...] = (),
) -> dict[str, float]:
    """Cross-dataset mean our-model/ablation metrics and their delta for one ablation.

    Args:
        ablation_name: Ablation label (the row key).
        per_dataset: ``{dataset: {"our_model": record, "ablation": record}}``.
        exclude: Dataset names dropped from the means (matching the averaged plots).

    Returns:
        A flat row with ``our_model_<m>``, ``ablation_<m>``, ``delta_<m>`` per metric.
    """
    row: dict[str, float] = {"ablation": ablation_name}
    records = [v for k, v in per_dataset.items() if k not in exclude]
    for m in METRICS:
        our_model = mean([r["our_model"].get(m, float("nan")) for r in records])
        ablation = mean([r["ablation"].get(m, float("nan")) for r in records])
        row[f"our_model_{m}"] = our_model
        row[f"ablation_{m}"] = ablation
        row[f"delta_{m}"] = ablation - our_model
    return row


def save_summary_csv(rows: list[dict[str, float]], save_dir: Path = TABLES_DIR) -> Path:
    """Write one cross-dataset summary row per comparison ablation.

    Args:
        rows: Rows from :func:`summary_row`.
        save_dir: Directory to write the CSV into.

    Returns:
        Path: The written ``summary.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "summary.csv"
    fieldnames = ["ablation"] + [
        f"{side}_{m}" for m in METRICS for side in ("our_model", "ablation", "delta")
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
