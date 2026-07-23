import argparse
import os
from datetime import datetime
from pathlib import Path

import dedicated_prior
import factor_contributions
import harness
import label_window
import learned_weights
import no_ckd_order
import plots
import tables
import timing
import wandb
from dotenv import load_dotenv

from pipeline.datasets import DEFAULT_SPLIT_SEED
from pipeline.train import Trainer

PROJECT = "cosi-betadecay"
# Split seeds per ablation (DEFAULT_SPLIT_SEED..+N-1): the standard run measures the
# across-seed mean ± std of every metric; the first seed alone reproduces the
# single-split pipeline numbers.
N_SEEDS = 10
# One timestamped run folder per invocation, mirroring the pipeline's results/
# convention: ablations/results/<timestamp>/{figures,tables}.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Comparison ablations (reference-vs-ablation metric plots). factor_contributions,
# label_window, and dedicated_prior are handled separately: a per-factor decomposition, a
# label-window sweep, and the dedicated-prior operating-point run (the former
# batch_analysis.py) respectively — none a single reference-vs-ablation comparison.
COMPARISON_ABLATIONS = ("learned_weights", "no_ckd_order")
ALL_ABLATIONS = (*COMPARISON_ABLATIONS, "factor_contributions", "label_window", "dedicated_prior", "timing")


# Ablations that tune their own decision threshold for F1 (the learned weights). For
# these the fair reference is our model at ITS best-F1 threshold too — not the deployed
# count-prior point, which is not F1-optimized and would hand an F1-tuned baseline an
# unfair edge despite equal ranking power (AUC).
F1_TUNED_ABLATIONS = {"learned_weights"}


def run_comparison(
    name: str,
    ds: harness.DatasetPaths,
    train: dict,
    eval_data: dict,
    config: dict,
    reference_eval: dict,
    reference_trainer: Trainer,
    prior_counts: dict,
    plots_dir: Path | None,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, dict]:
    """Dispatch one comparison ablation and pair it with the right our-model reference.

    Args:
        name: Comparison ablation name (one of :data:`COMPARISON_ABLATIONS`).
        ds: Dataset paths (used by no_ckd_order to re-extract energy-ordered features).
        train: Reference (CKD) training features.
        eval_data: Reference (CKD) eval features.
        config: The reference ``Trainer`` config, reused so only the ablation differs.
        reference_eval: The reference's deployed eval record (reference for deployed-point
            ablations like no_ckd_order).
        reference_trainer: The fitted reference, used to build the best-F1 reference for
            the F1-tuned ablations (so the threshold basis matches).
        prior_counts: The dataset's dedicated-prior counts, handed to the ablations
            decided at the deployed rule so they share the reference's operating point.
        plots_dir: Directory for this variant's diagnostic plots (density terms,
            eval confusion matrix, ROC/PR) — comparable figure-by-figure against
            the dataset's ``reference/`` set — or None to skip them (the extra
            seeds of a multi-seed run).
        seed: Split seed, so no_ckd_order's energy-ordered re-split pairs with the
            reference split it is compared against.

    Returns:
        ``{"our_model": <reference record>, "ablation": <ablation record>}``.

    Raises:
        ValueError: If ``name`` is not a known comparison ablation.
    """
    if name == "learned_weights":
        ablation = learned_weights.run(train, eval_data, config, plots_dir=plots_dir)
    elif name == "no_ckd_order":
        train_energy, eval_energy = harness.extract_split(ds, ordering="energy", seed=seed)
        ablation = no_ckd_order.run(train_energy, eval_energy, config, prior_counts, plots_dir=plots_dir)
    else:
        raise ValueError(f"Unknown comparison ablation: {name}")
    # Match the operating-point basis: best-F1 vs best-F1 for F1-tuned ablations,
    # deployed vs deployed otherwise.
    our_model = (
        harness.best_f1_record(reference_trainer, train, eval_data)
        if name in F1_TUNED_ABLATIONS
        else reference_eval
    )
    return {"our_model": our_model, "ablation": ablation}


def main(
    use_wandb: bool,
    dataset_filter: list[str] | None,
    ablation_filter: list[str] | None,
    exclude_avg: tuple[str, ...],
    n_seeds: int = N_SEEDS,
) -> None:
    """Run the selected ablations over every dataset and write figures + CSVs.

    Every ablation is repeated over the split seeds ``DEFAULT_SPLIT_SEED ..
    DEFAULT_SPLIT_SEED + n_seeds - 1`` on the same cached extraction, and the CSVs
    report the across-seed mean of each metric plus a ``<metric>_std`` spread
    column (NaN when ``n_seeds`` is 1). Reference and ablation share the split at
    each seed, so the summary deltas are paired (split noise cancels in the
    delta). Diagnostic figures and the dedicated_prior sky image are written for the
    first seed only — it reproduces the single-split pipeline numbers exactly.

    Args:
        use_wandb: Whether to log the figures to Weights & Biases.
        dataset_filter: Restrict to these dataset names (None = all discovered).
        ablation_filter: Restrict to these ablations (None = all of :data:`ALL_ABLATIONS`).
        exclude_avg: Dataset names dropped from the averaged plots and CSV means
            (default: none); they remain as per-dataset CSV rows.
        n_seeds: Number of split seeds per ablation (default :data:`N_SEEDS`).

    Raises:
        RuntimeError: If no datasets match the filter.
    """
    datasets = harness.discover_datasets()
    if dataset_filter:
        datasets = [d for d in datasets if d["name"] in dataset_filter]
    if not datasets:
        raise RuntimeError("No matching dataset folders found in data/.")
    selected = set(ablation_filter or ALL_ABLATIONS)
    seeds = tuple(DEFAULT_SPLIT_SEED + i for i in range(n_seeds))

    # This run's folder: ablations/results/<timestamp>/{figures,tables}, created
    # up front so the per-dataset, per-variant diagnostic plots land under
    # figures/<dataset>/{reference,<ablation>}/ as the loop runs.
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"

    if use_wandb:
        wandb.init(project=PROJECT, name="ablations")

    # Per-seed record lists, aggregated into mean ± std records after the loop.
    comparison = {name: {} for name in COMPARISON_ABLATIONS if name in selected}
    contributions: dict[str, list] = {}
    full_model: dict[str, list] = {}
    label_window_results: dict[str, list] = {}
    label_window_heuristic: dict[str, dict] = {}
    dedicated_prior_results: dict[str, dict] = {}
    timing_results: dict[str, dict] = {}

    for ds in datasets:
        name = ds["name"]
        print(f"\n{'=' * 60}\nExtracting + fitting reference: {name}\n{'=' * 60}")
        # Split-independent, so counted once per dataset and reused across seeds.
        prior_counts = harness.dedicated_prior_counts(ds)
        if "timing" in selected:
            # Seed-independent and cache-bypassing (it times the extraction itself),
            # so it runs once per dataset, outside the seed loop.
            print("  ablation: timing (uncached extraction, then fit + scoring medians)")
            timing_results[name] = timing.run(ds)
        for seed in seeds:
            first_seed = seed == seeds[0]
            if len(seeds) > 1:
                print(f"  seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})")
            train, eval_data = harness.extract_split(ds, ordering="ckd", seed=seed)
            reference_trainer, config = harness.reference(train, prior_counts)
            reference_eval = harness.metric_record(reference_trainer, eval_data)
            if first_seed:
                # The reference's own plot set, the reference every variant's
                # figures are compared against.
                harness.save_model_plots(reference_trainer, eval_data, figures_dir / name / "reference")

            for ablation_name in comparison:
                print(f"  ablation: {ablation_name}")
                comparison[ablation_name].setdefault(name, []).append(
                    run_comparison(
                        ablation_name,
                        ds,
                        train,
                        eval_data,
                        config,
                        reference_eval,
                        reference_trainer,
                        prior_counts,
                        plots_dir=figures_dir / name / ablation_name if first_seed else None,
                        seed=seed,
                    )
                )

            if "factor_contributions" in selected:
                print("  ablation: factor_contributions")
                contributions.setdefault(name, []).append(factor_contributions.run(train, eval_data))
                full_model.setdefault(name, []).append(
                    harness.bucket_full_scores(reference_trainer, eval_data)
                )

            if "label_window" in selected:
                print("  ablation: label_window (one pool extraction, relabeled per n_std)")
                label_window_results.setdefault(name, []).append(label_window.run(ds, seed=seed))
                if first_seed:
                    # Seed-independent (fit-free, whole-.sim): compute once per dataset.
                    print("  ablation: label_window 511 keV heuristic (one .sim pass, all windows)")
                    label_window_heuristic[name] = label_window.run_heuristic(ds)

            if "dedicated_prior" in selected:
                print("  dedicated_prior: reference at the dedicated-prior operating point")
                previous = dedicated_prior_results.get(name)
                record = dedicated_prior.run(
                    ds,
                    train,
                    eval_data,
                    plots_dir=figures_dir / name / "dedicated_prior" if first_seed else None,
                    prior_counts=prior_counts,
                )
                if previous is None:
                    dedicated_prior_results[name] = {
                        **record,
                        "train": [record["train"]],
                        "eval": [record["eval"]],
                    }
                else:
                    previous["train"].append(record["train"])
                    previous["eval"].append(record["eval"])

    figures, summary_rows = [], []
    for ablation_name, per_dataset in comparison.items():
        if not per_dataset:
            continue
        aggregated = {
            name: {
                side: harness.aggregate_seed_records([r[side] for r in recs])
                for side in ("our_model", "ablation")
            }
            for name, recs in per_dataset.items()
        }
        tables.save_comparison_csv(ablation_name, aggregated, tables_dir, exclude_avg)
        figures.append(plots.plot_metric_comparison(ablation_name, aggregated, figures_dir, exclude_avg))
        summary_rows.append(tables.summary_row(ablation_name, aggregated, exclude_avg, per_dataset))
    if summary_rows:
        tables.save_summary_csv(summary_rows, tables_dir)
    if contributions:
        contributions_agg = {
            name: {factor: harness.aggregate_seed_records([r[factor] for r in recs]) for factor in recs[0]}
            for name, recs in contributions.items()
        }
        full_model_agg = {name: harness.aggregate_seed_records(recs) for name, recs in full_model.items()}
        tables.save_factor_contributions_csv(contributions_agg, full_model_agg, tables_dir)
        figures.append(
            plots.plot_factor_contributions(contributions_agg, full_model_agg, figures_dir, exclude_avg)
        )
    if label_window_results:
        label_window_agg = {
            name: {n_std: harness.aggregate_seed_records([r[n_std] for r in recs]) for n_std in recs[0]}
            for name, recs in label_window_results.items()
        }
        tables.save_label_window_csv(label_window_agg, label_window_heuristic, tables_dir, exclude_avg)
        # Plot stays pure BEvAn — the heuristic is a CSV-only baseline.
        figures.append(plots.plot_label_window(label_window_agg, figures_dir, exclude_avg))
    if timing_results:
        tables.save_timing_csv(timing_results, timing.environment(), tables_dir)
    if dedicated_prior_results:
        dedicated_prior_agg = {
            name: {
                **rec,
                "train": harness.aggregate_seed_records(rec["train"]),
                "eval": harness.aggregate_seed_records(rec["eval"]),
            }
            for name, rec in dedicated_prior_results.items()
        }
        tables.save_dedicated_prior_csv(dedicated_prior_agg, tables_dir)

    if use_wandb:
        for path in figures:
            wandb.log({path.stem: wandb.Image(str(path))})
        wandb.finish()
    print(f"\nWrote {len(figures)} figure(s) and CSVs to {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse the driver CLI: logging and the dataset/ablation filters.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the β⁺ ablation study over every dataset in data/.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict to these dataset names")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        choices=ALL_ABLATIONS,
        help="Restrict to these ablations (default: all)",
    )
    parser.add_argument(
        "--exclude-avg",
        nargs="+",
        default=[],
        help="Datasets excluded from the averaged plots and CSV means (still written as per-dataset rows)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
    main(
        args.wandb,
        args.datasets,
        args.ablations,
        tuple(args.exclude_avg),
    )
