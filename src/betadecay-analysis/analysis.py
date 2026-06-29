import argparse
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
from utils.local_results import (
    save_confusion_matrix,
    save_density_terms,
    save_score_curves,
    write_metrics_json,
)
from utils.reader_extraction import get_reader
from utils.wandb_logging import log_density_terms

PROJECT = "cosi-betadecay"
RESULTS_DIR = Path("results")


def main(geo_file: str, sim_file: str, tra_file: str, use_wandb: bool, n_iter: int, calibrate: bool) -> None:
    """Run the β⁺ annihilation detection pipeline on a given MEGAlib dataset.

    Args:
        geo_file (str): Path to the MEGAlib geometry setup file.
        sim_file (str): Path to the MEGAlib .sim file containing simulated events.
        tra_file (str): Path to the MEGAlib .tra file containing Compton events.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        n_iter (int): Champion-seeded bin-search candidates (0 keeps the champion seed).
        calibrate (bool): Whether to calibrate the decision-threshold offset for F1.

    Raises:
        ValueError: If any of the input files have the wrong extension.
        FileNotFoundError: If any of the input files do not exist.
    """
    sim_path, tra_path = Path(sim_file), Path(tra_file)
    if sim_path.suffix != ".sim":
        raise ValueError(f"--sim-file must be a .sim file, got: {sim_file}")
    if tra_path.suffix != ".tra":
        raise ValueError(f"--tra-file must be a .tra file, got: {tra_file}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")
    if not tra_path.is_file():
        raise FileNotFoundError(f"Tra file not found: {tra_file}")

    if use_wandb:
        wandb.init(project=PROJECT)

    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(tra_file)
    print(f"Accumulated {n_events} Compton events from {tra_file}")
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(geo_file, sim_file)

    datasets = Datasets(reader_sim, unit_vector)
    data = datasets.compute_event_features()
    train, eval_data = datasets.split_dataset(data)

    # Champion-seeded, AUC-rewarded search for bins + smoothing on the training
    # split (overfit-guarded: keeps the champion unless beaten by >1 SE), then
    # refit on the full training set at the chosen config.
    best_config, info = search_hyperparams(train, n_iter=n_iter)
    flat = flatten_config(best_config)
    print(
        f"Selected {flat} (accepted={info['accepted']}; confirm AUC "
        f"{info['confirm_best_mean']:.4f} vs seed {info['confirm_seed_mean']:.4f})"
    )
    selection = {**flat, "selection_accepted": info["accepted"], "cv_auc": info["confirm_best_mean"]}
    if use_wandb:
        wandb.log(selection)

    trainer = Trainer(**best_config).fit(train)

    # Calibrate the global decision-threshold offset for F1 on the training split
    # (guarded; a 0 offset keeps the count-prior default), then deploy it.
    if calibrate:
        offset, cal_info = calibrate_global_offset(trainer, train, best_config)
        apply_offset(trainer, offset)
        print(
            f"Threshold offset c={offset:.4f} (accepted={cal_info['accepted']}; "
            f"CV F1 gain {cal_info.get('cv_mean_gain', float('nan')):.4f})"
        )
        selection["threshold_offset"] = offset
        selection["threshold_moved"] = cal_info["accepted"]
        if use_wandb:
            wandb.log({"threshold_offset": offset, "threshold_moved": cal_info["accepted"]})
    else:
        selection["threshold_offset"] = 0.0
        selection["threshold_moved"] = False

    # Per-run results folder: results/<timestamp>/<dataset>/ (same convention the
    # batch CSV uses), created up front like batch_analysis so results/ always
    # exists even if a step writes nothing. Density terms are model-level, so log
    # + save them once.
    name = Path(sim_file).stem
    out_dir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_density_terms(trainer)
    save_density_terms(trainer, out_dir)

    evaluator = Evaluator(trainer)
    split_metrics = {}
    for split_name, split_data in (("train", train), ("eval", eval_data)):
        counts = evaluator.evaluate(split_data, split_name=split_name)
        split_metrics[split_name] = {**counts, **metrics(counts), **prior_free_scores(evaluator, split_data)}
        save_confusion_matrix(counts, out_dir, split_name)
        save_score_curves(evaluator, split_data, out_dir, split_name)

    # One metrics.json per dataset, both splits, with the chosen config.
    path = write_metrics_json(
        out_dir,
        {
            "dataset": name,
            "geo_file": geo_file,
            "sim_file": sim_file,
            "tra_file": tra_file,
            "selection": selection,
            "train": split_metrics["train"],
            "eval": split_metrics["eval"],
        },
    )
    print(f"Saved results to {path.parent.resolve()}")

    if use_wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: the geo/sim/tra file paths and the ``--wandb`` flag.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run β⁺ annihilation detection on one .sim/.tra dataset.")
    parser.add_argument(
        "--geo-file",
        default="$MEGALIB/resource/examples/geomega/special/SPILike.geo.setup",
        help="MEGAlib geometry setup file",
    )
    parser.add_argument("--sim-file", default="data/SPILike.sim", help="MEGAlib .sim file (simulated events)")
    parser.add_argument("--tra-file", default="data/SPILike.tra", help="MEGAlib .tra file (Compton events)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--search-iters",
        type=int,
        default=50,
        help="Champion-seeded bin-search candidates (0 keeps the champion seed; cost scales with this x folds)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the decision-threshold calibration (keep the raw count-prior operating point)",
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
    main(args.geo_file, args.sim_file, args.tra_file, args.wandb, args.search_iters, not args.no_calibrate)
