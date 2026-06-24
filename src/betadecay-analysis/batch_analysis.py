import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv

from dataset.datasets import Datasets
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator, metrics, prior_free_scores
from pipeline.model_selection import apply_offset, calibrate_global_offset, flatten_config, search_hyperparams
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

PROJECT = "cosi-betadecay"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Static dataset-name -> geometry setup map for the batch runner. Each simulation
# in data/ was produced against a different geometry, so the batch runner cannot
# take a single --geo-file flag; the per-dataset entry points (analysis.py,
# train_model.py, inference.py) pass geometry explicitly instead.
GEOMETRIES = {
    "SPILike": "$MEGALIB/resource/examples/geomega/special/SPILike.geo.setup",
    "NCT": "$MEGALIB/resource/examples/geomega/special/Simple_Ge_NCT_try1_040721.geo.setup",
    "Max": "$MEGALIB/resource/examples/geomega/special/Max.geo.setup",
    "GeACT": "$MEGALIB/resource/examples/geomega/special/GeACT.geo.setup",
    "COSI_Balloon": "$MEGALIB/resource/examples/geomega/cosiballoon/COSIBalloon.12Detector.geo.setup",
}


def discover_datasets() -> list[dict[str, str]]:
    """Find every {name}.sim with a matching {name}.tra and a registered geometry.

    A simulation is skipped (with a message) if it has no matching ``.tra`` or no
    entry in :data:`GEOMETRIES`.

    Returns:
        list[dict[str, str]]: One dict per dataset with name, geo_file,
            sim_file, and tra_file keys.
    """
    datasets = []
    for sim_file in sorted(DATA_DIR.glob("*.sim")):
        tra_file = sim_file.with_suffix(".tra")
        if not tra_file.exists():
            print(f"Skipping {sim_file.name}: no matching {tra_file.name}")
            continue
        name = sim_file.stem
        geo_file = GEOMETRIES.get(name)
        if geo_file is None:
            print(f"Skipping {name}: no geometry registered in GEOMETRIES")
            continue
        datasets.append(
            {
                "name": name,
                "geo_file": geo_file,
                "sim_file": str(sim_file),
                "tra_file": str(tra_file),
            }
        )
    return datasets


def run_one(ds: dict[str, str], use_wandb: bool, n_iter: int, calibrate: bool) -> tuple[dict, dict]:
    """Run the exact analysis.py pipeline for a single (geo, sim, tra) dataset.

    Args:
        ds (dict[str, str]): Dataset paths with name, geo_file, sim_file, tra_file.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        n_iter (int): Number of champion-seeded search candidates per dataset.
        calibrate (bool): Whether to calibrate the decision-threshold offset for F1.

    Returns:
        tuple[dict, dict]: (per-split confusion counts plus prior-free metrics,
            the flattened chosen config plus selection diagnostics).
    """
    if use_wandb:
        wandb.init(project=PROJECT, name=ds["name"], reinit=True)

    imager = FarFieldImager(
        geometry_file=ds["geo_file"],
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(ds["tra_file"])
    print(f"Accumulated {n_events} Compton events from {ds['tra_file']}")
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(ds["geo_file"], ds["sim_file"])

    datasets = Datasets(reader_sim, unit_vector)
    data = datasets.compute_event_features()
    train, eval_data = datasets.split_dataset(data)

    # Champion-seeded, AUC-rewarded search per dataset (geometries differ wildly
    # in event counts), leak-free on the training split and overfit-guarded (keeps
    # the champion unless beaten by >1 SE), then refit on the full train set.
    best_config, info = search_hyperparams(train, n_iter=n_iter)
    flat = flatten_config(best_config)
    print(
        f"  selected {flat} (accepted={info['accepted']}; confirm AUC "
        f"{info['confirm_best_mean']:.4f} vs seed {info['confirm_seed_mean']:.4f})"
    )
    selection = {**flat, "selection_accepted": info["accepted"], "cv_auc": info["confirm_best_mean"]}

    trainer = Trainer(**best_config).fit(train)

    # Calibrate the global decision-threshold offset for F1 (guarded; 0 keeps the
    # count-prior default and stays byte-identical), then deploy it.
    if calibrate:
        offset, cal_info = calibrate_global_offset(trainer, train, best_config)
        apply_offset(trainer, offset)
        print(f"  threshold offset c={offset:.4f} (accepted={cal_info['accepted']})")
        selection["threshold_offset"] = offset
        selection["threshold_moved"] = cal_info["accepted"]
    else:
        selection["threshold_offset"] = 0.0
        selection["threshold_moved"] = False
    if use_wandb:
        wandb.log(selection)

    evaluator = Evaluator(trainer)
    results = {}
    for split_name, split_data in (("train", train), ("eval", eval_data)):
        counts = evaluator.evaluate(split_data, split_name=split_name)
        results[split_name] = {**counts, **prior_free_scores(evaluator, split_data)}

    if use_wandb:
        wandb.finish()
    return results, selection


def main(use_wandb: bool, n_iter: int, calibrate: bool) -> None:
    """Run annihilation detection extraction over every .sim/.tra/geometry triple.

    Args:
        use_wandb (bool): Whether to use Weights & Biases for logging.
        n_iter (int): Number of champion-seeded search candidates per dataset.
        calibrate (bool): Whether to calibrate the decision-threshold offset for F1.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        RuntimeError: If the data directory contains no ``.sim``/``.tra`` pairs.
    """
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR.resolve()}")
    datasets = discover_datasets()
    if not datasets:
        raise RuntimeError(f"No .sim/.tra pairs found in {DATA_DIR.resolve()}")

    print(f"Found {len(datasets)} dataset(s): {', '.join(d['name'] for d in datasets)}")
    rows = []
    for ds in datasets:
        print(
            f"\n{'=' * 60}\nRunning {ds['name']}\n  geo: {ds['geo_file']}\n  sim: {ds['sim_file']}\n  tra: {ds['tra_file']}\n{'=' * 60}"
        )
        totals, selection = run_one(ds, use_wandb, n_iter, calibrate)
        for split, counts in totals.items():
            rows.append(
                {
                    "dataset": ds["name"],
                    "split": split,
                    "geo_file": ds["geo_file"],
                    "sim_file": ds["sim_file"],
                    "tra_file": ds["tra_file"],
                    **{k: counts[k] for k in ("tp", "fp", "fn", "tn", "excluded")},
                    **metrics(counts),
                    # Prior-free, threshold-independent separating power: best_f1 is
                    # the best F1 over all score thresholds (ignores the prior),
                    # auc is the ROC AUC of the per-event likelihood ratio.
                    "best_f1": counts["best_f1"],
                    "auc": counts["auc"],
                    # The per-dataset config the search chose, for traceability.
                    **selection,
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = RESULTS_DIR / f"{timestamp}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results for {len(rows)} dataset(s) to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run β⁺ detection over every .sim/.tra pair found in data/.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--search-iters",
        type=int,
        default=50,
        help="Champion-seeded search candidates per dataset (0 keeps the champion seed; cost scales with this x folds x datasets)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the decision-threshold calibration (keep the raw count-prior operating point)",
    )
    args = parser.parse_args()

    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)

    main(args.wandb, args.search_iters, not args.no_calibrate)
