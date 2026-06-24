import argparse
import os
from datetime import datetime
from pathlib import Path

import factor_contributions
import harness
import learned_weights
import no_calibration
import no_ckd_order
import plots
import tables
import wandb
from dotenv import load_dotenv

PROJECT = "cosi-betadecay"
# One timestamped run folder per invocation, mirroring the pipeline's results/
# convention: ablations/results/<timestamp>/{figures,tables}.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Comparison ablations (champion-vs-ablation metric plots). factor_contributions is
# handled separately: its output is a per-factor decomposition, not a comparison.
COMPARISON_ABLATIONS = ("no_calibration", "learned_weights", "no_ckd_order")
ALL_ABLATIONS = (*COMPARISON_ABLATIONS, "factor_contributions")


# Ablations that tune their own decision threshold for F1 (the learned weights). For
# these the fair reference is our model at ITS best-F1 threshold too — not the deployed
# count-prior point, which is not F1-optimized and would hand an F1-tuned baseline an
# unfair edge despite equal ranking power (AUC).
F1_TUNED_ABLATIONS = {"learned_weights"}


def run_comparison(
    name: str,
    ds: dict[str, str],
    train: dict,
    eval_data: dict,
    config: dict,
    champion_eval: dict,
    champion_trainer: object,
) -> dict[str, dict]:
    """Dispatch one comparison ablation and pair it with the right our-model reference.

    Args:
        name: Comparison ablation name (one of :data:`COMPARISON_ABLATIONS`).
        ds: Dataset paths (used by no_ckd_order to re-extract energy-ordered features).
        train: Champion (CKD) training features.
        eval_data: Champion (CKD) eval features.
        config: The champion ``Trainer`` config, reused so only the ablation differs.
        champion_eval: The champion's deployed eval record (reference for deployed-point
            ablations like no_calibration / no_ckd_order).
        champion_trainer: The fitted champion, used to build the best-F1 reference for
            the F1-tuned ablations (so the threshold basis matches).

    Returns:
        ``{"our_model": <reference record>, "ablation": <ablation record>}``.

    Raises:
        ValueError: If ``name`` is not a known comparison ablation.
    """
    if name == "no_calibration":
        ablation = no_calibration.run(train, eval_data, config)
    elif name == "learned_weights":
        ablation = learned_weights.run(train, eval_data, config)
    elif name == "no_ckd_order":
        train_energy, eval_energy = harness.extract_split(ds, ordering="energy")
        ablation = no_ckd_order.run(train_energy, eval_energy, config)
    else:
        raise ValueError(f"Unknown comparison ablation: {name}")
    # Match the operating-point basis: best-F1 vs best-F1 for F1-tuned ablations,
    # deployed vs deployed otherwise.
    our_model = (
        harness.best_f1_record(champion_trainer, train, eval_data)
        if name in F1_TUNED_ABLATIONS
        else champion_eval
    )
    return {"our_model": our_model, "ablation": ablation}


def main(
    use_wandb: bool,
    n_iter: int,
    calibrate: bool,
    dataset_filter: list[str] | None,
    ablation_filter: list[str] | None,
    exclude_avg: tuple[str, ...],
) -> None:
    """Run the selected ablations over every dataset and write figures + CSVs.

    Args:
        use_wandb: Whether to log the figures to Weights & Biases.
        n_iter: Champion-seeded search candidates per dataset.
        calibrate: Whether the champion calibrates its threshold offset.
        dataset_filter: Restrict to these dataset names (None = all discovered).
        ablation_filter: Restrict to these ablations (None = all of :data:`ALL_ABLATIONS`).
        exclude_avg: Dataset names dropped from the averaged plots and CSV means
            (e.g. the out-of-domain balloon); they remain as per-dataset CSV rows.

    Raises:
        RuntimeError: If no datasets match the filter.
    """
    datasets = harness.discover_datasets()
    if dataset_filter:
        datasets = [d for d in datasets if d["name"] in dataset_filter]
    if not datasets:
        raise RuntimeError("No matching .sim/.tra datasets found in data/.")
    selected = set(ablation_filter or ALL_ABLATIONS)

    if use_wandb:
        wandb.init(project=PROJECT, name="ablations")

    comparison = {name: {} for name in COMPARISON_ABLATIONS if name in selected}
    contributions: dict[str, dict] = {}
    full_model: dict[str, dict] = {}

    for ds in datasets:
        name = ds["name"]
        print(f"\n{'=' * 60}\nExtracting + fitting champion: {name}\n{'=' * 60}")
        train, eval_data = harness.extract_split(ds, ordering="ckd")
        champion_trainer, config = harness.champion(train, n_iter=n_iter, calibrate=calibrate)
        champion_eval = harness.metric_record(champion_trainer, eval_data)

        for ablation_name in comparison:
            print(f"  ablation: {ablation_name}")
            comparison[ablation_name][name] = run_comparison(
                ablation_name, ds, train, eval_data, config, champion_eval, champion_trainer
            )

        if "factor_contributions" in selected:
            print("  ablation: factor_contributions")
            contributions[name] = factor_contributions.run(train, eval_data)
            full_model[name] = harness.bucket_full_scores(champion_trainer, eval_data)

    # This run's folder: ablations/results/<timestamp>/{figures,tables}.
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"

    figures, summary_rows = [], []
    for ablation_name, per_dataset in comparison.items():
        if not per_dataset:
            continue
        tables.save_comparison_csv(ablation_name, per_dataset, tables_dir, exclude_avg)
        figures.append(plots.plot_metric_comparison(ablation_name, per_dataset, figures_dir, exclude_avg))
        summary_rows.append(tables.summary_row(ablation_name, per_dataset, exclude_avg))
    if summary_rows:
        tables.save_summary_csv(summary_rows, tables_dir)
    if contributions:
        tables.save_factor_contributions_csv(contributions, full_model, tables_dir)
        figures.append(plots.plot_factor_contributions(contributions, full_model, figures_dir, exclude_avg))

    if use_wandb:
        for path in figures:
            wandb.log({path.stem: wandb.Image(str(path))})
        wandb.finish()
    print(f"\nWrote {len(figures)} figure(s) and CSVs to {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse the driver CLI: logging, search budget, calibration, and the filters.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the β⁺ ablation study over every dataset in data/.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--search-iters",
        type=int,
        default=50,
        help="Champion-seeded search candidates per dataset (0 keeps the champion seed)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the champion's decision-threshold calibration",
    )
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
        default=["COSI_Balloon"],
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
        args.search_iters,
        not args.no_calibrate,
        args.datasets,
        args.ablations,
        tuple(args.exclude_avg),
    )
