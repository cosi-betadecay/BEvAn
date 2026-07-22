import csv
import math
from pathlib import Path

# Default output directory: ablations/tables (created on first write).
TABLES_DIR = Path(__file__).resolve().parent / "tables"

# The metric keys shared across the comparison CSVs (record keys from the pipeline).
METRICS = ("f1_score", "auc", "precision", "recall", "fpr")
# The 511 keV baseline's columns in the label-window CSV: a hard cut has no ranked
# score (no auc) and is deterministic (no seed spread), so only these four.
HEURISTIC_METRICS = ("f1_score", "precision", "recall", "fpr")
# Ablation-specific scalar columns, written on the ablation side only since the
# reference has no counterpart to them. The learned-weights entry mirrors
# ``harness.FACTORS``, spelled out rather than imported because harness pulls in
# ROOT and this module stays importable without MEGAlib.
EXTRA_ABLATION_METRICS: dict[str, tuple[str, ...]] = {
    "learned_weights": ("w_delta_E", "w_arm", "w_anni", "bias"),
}


def mean(values: list[float]) -> float:
    """Mean of the finite entries (NaN if there are none)."""
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def std(values: list[float]) -> float:
    """Population std of the finite entries (NaN if fewer than two)."""
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2:
        return float("nan")
    m = sum(finite) / len(finite)
    return math.sqrt(sum((v - m) ** 2 for v in finite) / len(finite))


def save_comparison_csv(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float | int | dict[str, float]]]],
    save_dir: Path = TABLES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Write one comparison ablation's per-dataset metrics, with a trailing mean row.

    Columns are ``dataset`` then ``our_model_<m>`` / ``ablation_<m>`` for each metric
    in :data:`METRICS`, followed by the matching ``<side>_<m>_std`` seed-spread
    columns (NaN for single-seed runs). Ablations listed in
    :data:`EXTRA_ABLATION_METRICS` append their own ``ablation_<key>`` columns and
    spreads — the learned-weights fit records the weights it found, so the quoted
    values trace to the run rather than to a refit. Every dataset gets a row; only
    the trailing mean excludes ``exclude`` (so the mean matches the averaged plot
    while the detail stays whole). The mean row's std columns hold the mean of the
    finite per-dataset spreads — the typical seed noise, not a spread of spreads.

    Args:
        ablation_name: Ablation label and output filename stem.
        per_dataset: ``{dataset: {"our_model": record, "ablation": record}}`` with
            seed-aggregated records (``<m>`` mean keys plus ``<m>_std`` spreads).
        save_dir: Directory to write the CSV into.
        exclude: Dataset names dropped from the mean row only (kept as detail rows).

    Returns:
        Path: The written ``<ablation_name>.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{ablation_name}.csv"
    columns = [f"{side}_{m}" for m in METRICS for side in ("our_model", "ablation")]
    extras = EXTRA_ABLATION_METRICS.get(ablation_name, ())
    extra_columns = [f"ablation_{e}" for e in extras]
    fieldnames = [
        "dataset",
        *columns,
        *(f"{c}_std" for c in columns),
        *extra_columns,
        *(f"{c}_std" for c in extra_columns),
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, rec in per_dataset.items():
            row = {"dataset": dataset}
            for m in METRICS:
                for side in ("our_model", "ablation"):
                    row[f"{side}_{m}"] = rec[side].get(m, float("nan"))
                    row[f"{side}_{m}_std"] = rec[side].get(f"{m}_std", float("nan"))
            for e in extras:
                row[f"ablation_{e}"] = rec["ablation"].get(e, float("nan"))
                row[f"ablation_{e}_std"] = rec["ablation"].get(f"{e}_std", float("nan"))
            writer.writerow(row)
        included = [r for k, r in per_dataset.items() if k not in exclude]
        mean_row = {"dataset": f"mean (excl. {', '.join(exclude)})" if exclude else "mean"}
        for m in METRICS:
            for side in ("our_model", "ablation"):
                mean_row[f"{side}_{m}"] = mean([r[side].get(m, float("nan")) for r in included])
                mean_row[f"{side}_{m}_std"] = mean([r[side].get(f"{m}_std", float("nan")) for r in included])
        for e in extras:
            mean_row[f"ablation_{e}"] = mean([r["ablation"].get(e, float("nan")) for r in included])
            mean_row[f"ablation_{e}_std"] = mean(
                [r["ablation"].get(f"{e}_std", float("nan")) for r in included]
            )
        writer.writerow(mean_row)
    return path


def save_factor_contributions_csv(
    contributions_by_dataset: dict[str, dict[str, dict[str, float]]],
    full_model_by_dataset: dict[str, dict[str, float]] | None = None,
    save_dir: Path = TABLES_DIR,
) -> Path:
    """Write per-dataset, per-factor F1/AUC in long format for pivoting.

    Columns: ``dataset``, ``factor``, ``f1``, ``auc``, the full-model ``full_f1`` /
    ``full_auc`` reference for that dataset, and the matching ``*_std`` seed-spread
    columns (NaN for single-seed runs).

    Args:
        contributions_by_dataset: ``{dataset: {factor: {"f1": ..., "auc": ...}}}``
            with seed-aggregated records (``f1_std``/``auc_std`` spreads).
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
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "factor",
                "f1",
                "auc",
                "full_f1",
                "full_auc",
                "f1_std",
                "auc_std",
                "full_f1_std",
                "full_auc_std",
            ],
        )
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
                        "f1_std": vals.get("f1_std", float("nan")),
                        "auc_std": vals.get("auc_std", float("nan")),
                        "full_f1_std": full.get("f1_std", float("nan")),
                        "full_auc_std": full.get("auc_std", float("nan")),
                    }
                )
    return path


def save_label_window_csv(
    results_by_dataset: dict[str, dict[int, dict[str, float | int]]],
    heuristic: dict[str, dict[int, dict[str, float | int]]],
    save_dir: Path = TABLES_DIR,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Write the label-window sweep in long format: one row per (dataset, n_std).

    Columns are ``dataset``, ``n_std``, the :data:`METRICS`, a ``<m>_std``
    seed-spread column per metric (NaN for single-seed runs), then the energy-only
    511 keV baseline's ``heuristic_<m>`` for ``m`` in :data:`HEURISTIC_METRICS` at
    the same ``n_std``. The baseline is always emitted — it is a fixed part of this
    table, not an option — but carries no ``auc`` (a hard cut has no ranked score)
    and no ``_std`` (it is fit-free and deterministic). Per-dataset rows keep every
    dataset; trailing ``mean`` rows (one per swept ``n_std``) average the
    non-``exclude`` datasets for both the model and the baseline, matching the
    averaged plot — the model's std columns hold the mean of the finite per-dataset
    spreads (the typical seed noise).

    Args:
        results_by_dataset: ``{dataset: {n_std: metric_record}}`` with
            seed-aggregated records (``<m>`` mean keys plus ``<m>_std`` spreads).
        heuristic: ``{dataset: {n_std: record}}`` for the 511 keV baseline; a
            dataset/window missing from it writes NaN baseline cells.
        save_dir: Directory to write the CSV into.
        exclude: Dataset names dropped from the mean rows only (kept as detail rows).

    Returns:
        Path: The written ``label_window.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "label_window.csv"
    n_std_values = sorted({n for per_n in results_by_dataset.values() for n in per_n})

    def heuristic_row(record: dict[str, float | int]) -> dict[str, float]:
        """The ``heuristic_<m>`` cells for one baseline record (NaN where absent)."""
        return {f"heuristic_{m}": record.get(m, float("nan")) for m in HEURISTIC_METRICS}

    with path.open("w", newline="") as fh:
        fieldnames = [
            "dataset",
            "n_std",
            *METRICS,
            *(f"{m}_std" for m in METRICS),
            *(f"heuristic_{m}" for m in HEURISTIC_METRICS),
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, per_n in results_by_dataset.items():
            for n_std in sorted(per_n):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "n_std": n_std,
                        **{m: per_n[n_std].get(m, float("nan")) for m in METRICS},
                        **{f"{m}_std": per_n[n_std].get(f"{m}_std", float("nan")) for m in METRICS},
                        **heuristic_row(heuristic.get(dataset, {}).get(n_std, {})),
                    }
                )
        label = f"mean (excl. {', '.join(exclude)})" if exclude else "mean"
        for n_std in n_std_values:
            included = [
                per_n[n_std] for k, per_n in results_by_dataset.items() if k not in exclude and n_std in per_n
            ]
            included_heur = [
                heuristic[k][n_std]
                for k in results_by_dataset
                if k not in exclude and n_std in heuristic.get(k, {})
            ]
            writer.writerow(
                {
                    "dataset": label,
                    "n_std": n_std,
                    **{m: mean([r.get(m, float("nan")) for r in included]) for m in METRICS},
                    **{
                        f"{m}_std": mean([r.get(f"{m}_std", float("nan")) for r in included]) for m in METRICS
                    },
                    **{
                        f"heuristic_{m}": mean([r.get(m, float("nan")) for r in included_heur])
                        for m in HEURISTIC_METRICS
                    },
                }
            )
    return path


def save_dedicated_prior_csv(
    results_by_dataset: dict[str, dict],
    save_dir: Path = TABLES_DIR,
) -> Path:
    """Write the deployed (dedicated-prior) record per dataset and split.

    The successor to ``batch_analysis.py``'s ``summary.csv``: one row per
    (dataset, split) with the confusion counts and metrics at the dedicated-prior
    operating point, ``<m>_std`` seed-spread columns per metric (NaN for
    single-seed runs; counts and metrics are across-seed means, so the counts are
    fractional there), plus the counted per-bucket P(β⁺) prior columns.

    Args:
        results_by_dataset: ``{dataset: dedicated_prior record}`` with seed-aggregated
            ``train``/``eval`` records (``<m>`` mean keys plus ``<m>_std`` spreads).
        save_dir: Directory to write the CSV into.

    Returns:
        Path: The written ``dedicated_prior.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "dedicated_prior.csv"
    counts = ("tp", "fp", "fn", "tn", "excluded")
    scored = (*METRICS, "best_f1")
    buckets = sorted({b for rec in results_by_dataset.values() for b in rec["priors"]})
    fieldnames = [
        "dataset",
        "split",
        *counts,
        *scored,
        *(f"{m}_std" for m in scored),
        *(f"prior_bucket_{b}" for b in buckets),
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, rec in results_by_dataset.items():
            for split in ("train", "eval"):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "split": split,
                        **{c: rec[split].get(c, float("nan")) for c in counts},
                        **{m: rec[split].get(m, float("nan")) for m in scored},
                        **{f"{m}_std": rec[split].get(f"{m}_std", float("nan")) for m in scored},
                        **{f"prior_bucket_{b}": rec["priors"].get(b, float("nan")) for b in buckets},
                    }
                )
    return path


def save_timing_csv(
    results_by_dataset: dict[str, dict],
    env: dict,
    save_dir: Path = TABLES_DIR,
) -> Path:
    """Write the per-stage runtime budget per dataset, with the environment appended.

    One row per dataset holding the extraction / fit / scoring costs, the artifact
    size and the density-cell count. The environment block is written as trailing
    ``# key: value`` comment lines rather than repeated columns: it is identical for
    every row and it is what makes a quoted figure traceable to a machine.

    Args:
        results_by_dataset: ``{dataset: timing.run record}``.
        env: The :func:`timing.environment` block for the run.
        save_dir: Directory to write the CSV into.

    Returns:
        Path: The written ``timing.csv``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "timing.csv"
    columns = (
        "n_events",
        "n_eval_events",
        "extract_s",
        "extract_us_per_event",
        "fit_s",
        "score_s",
        "score_events_per_s_batched",
        "extract_over_score_per_event",
        "artifact_kb",
        "n_density_cells",
    )

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", *columns])
        writer.writeheader()
        for dataset, rec in results_by_dataset.items():
            writer.writerow({"dataset": dataset, **{c: rec.get(c, float("nan")) for c in columns}})
        for key, value in env.items():
            fh.write(f"# {key}: {value}\n")
    return path


def summary_row(
    ablation_name: str,
    per_dataset: dict[str, dict[str, dict[str, float | int | dict[str, float]]]],
    exclude: tuple[str, ...] = (),
    per_dataset_seeds: dict[str, list[dict[str, dict]]] | None = None,
) -> dict[str, str | float]:
    """Cross-dataset mean our-model/ablation metrics and their delta for one ablation.

    Args:
        ablation_name: Ablation label (the row key).
        per_dataset: ``{dataset: {"our_model": record, "ablation": record}}`` with
            seed-aggregated records.
        exclude: Dataset names dropped from the means (matching the averaged plots).
        per_dataset_seeds: The unaggregated per-seed pairs,
            ``{dataset: [{"our_model": record, "ablation": record} per seed]}``.
            When given (and >1 seed), ``delta_<m>_std`` is the std across seeds of
            the cross-dataset mean *paired* delta — reference and ablation share the
            split at each seed, so the common split noise cancels in the delta and
            the spread measures only the effect's stability. NaN otherwise.

    Returns:
        A flat row with ``our_model_<m>``, ``ablation_<m>``, ``delta_<m>``, and
        ``delta_<m>_std`` per metric.
    """
    row: dict[str, str | float] = {"ablation": ablation_name}
    records = [v for k, v in per_dataset.items() if k not in exclude]
    seed_lists = [v for k, v in (per_dataset_seeds or {}).items() if k not in exclude]
    n_seeds = min((len(lst) for lst in seed_lists), default=0)
    for m in METRICS:
        our_model = mean([r["our_model"].get(m, float("nan")) for r in records])
        ablation = mean([r["ablation"].get(m, float("nan")) for r in records])
        row[f"our_model_{m}"] = our_model
        row[f"ablation_{m}"] = ablation
        row[f"delta_{m}"] = ablation - our_model
        per_seed_deltas = [
            mean(
                [
                    lst[s]["ablation"].get(m, float("nan")) - lst[s]["our_model"].get(m, float("nan"))
                    for lst in seed_lists
                ]
            )
            for s in range(n_seeds)
        ]
        row[f"delta_{m}_std"] = std(per_seed_deltas)
    return row


def save_summary_csv(rows: list[dict[str, str | float]], save_dir: Path = TABLES_DIR) -> Path:
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
        f"{side}_{m}{suffix}"
        for m in METRICS
        for side, suffix in (("our_model", ""), ("ablation", ""), ("delta", ""), ("delta", "_std"))
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
