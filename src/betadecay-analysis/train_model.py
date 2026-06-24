import argparse
import os
from pathlib import Path

import wandb
from dotenv import load_dotenv

from dataset.datasets import Datasets
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.model_selection import apply_offset, calibrate_global_offset, flatten_config, search_hyperparams
from pipeline.train import Trainer
from utils.reader_extraction import get_reader
from utils.wandb_logging import log_density_terms

PROJECT = "cosi-betadecay"
MODELS_DIR = Path("models")


def train_and_save(
    geo_file: str,
    sim_file: str,
    tra_file: str,
    out_file: str,
    use_wandb: bool,
    n_iter: int,
    calibrate: bool,
) -> None:
    """Fit the per-bucket model on a whole simulation and save it to a torch artifact.

    Unlike :mod:`analysis`, this fits on every event in the simulation (no
    train/eval split): held-out testing happens later on a separate simulation
    via :mod:`inference`. The champion-seeded, overfit-guarded search and the
    guarded threshold calibration are identical to the ``analysis.py`` selection,
    so the saved model's operating point matches what ``analysis.py`` would
    deploy. The fitted
    densities — with the calibrated ``log_threshold`` baked in — are written to
    ``out_file`` for later inference.

    Args:
        geo_file: MEGAlib geometry setup file.
        sim_file: MEGAlib ``.sim`` file (simulated events) to train on.
        tra_file: MEGAlib ``.tra`` file (Compton events) for source reconstruction.
        out_file: Destination path for the saved ``.pt`` model artifact.
        use_wandb: Whether to log selection diagnostics to Weights & Biases.
        n_iter: Champion-seeded search candidates (0 keeps the champion seed).
        calibrate: Whether to calibrate the decision-threshold offset for F1.

    Raises:
        ValueError: If ``sim_file``/``tra_file`` have the wrong extension.
        FileNotFoundError: If ``sim_file`` or ``tra_file`` does not exist.
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
        wandb.init(project=PROJECT, name=f"train:{sim_path.stem}")

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

    # Fit on the full simulation (no train/eval split): the held-out test set is a
    # separate simulation run through inference.py. The search is still leak-free
    # k-fold on this data and overfit-guarded, matching the analysis.py selection.
    best_config, info = search_hyperparams(data, n_iter=n_iter)
    flat = flatten_config(best_config)
    print(
        f"Selected {flat} (accepted={info['accepted']}; confirm AUC "
        f"{info['confirm_best_mean']:.4f} vs seed {info['confirm_seed_mean']:.4f})"
    )
    if use_wandb:
        wandb.log({**flat, "selection_accepted": info["accepted"], "cv_auc": info["confirm_best_mean"]})

    trainer = Trainer(**best_config).fit(data)

    # Calibrate the global decision-threshold offset for F1 (guarded; a 0 offset
    # keeps the count-prior default), then bake it into the model before saving.
    offset = 0.0
    if calibrate:
        offset, cal_info = calibrate_global_offset(trainer, data, best_config)
        apply_offset(trainer, offset)
        print(
            f"Threshold offset c={offset:.4f} (accepted={cal_info['accepted']}; "
            f"CV F1 gain {cal_info.get('cv_mean_gain', float('nan')):.4f})"
        )
        if use_wandb:
            wandb.log({"threshold_offset": offset, "threshold_moved": cal_info["accepted"]})

    # Log the fitted per-bucket density terms (model-level; no-op without a run).
    log_density_terms(trainer)

    trainer.save(
        out_file,
        threshold_offset=offset,
        provenance={"sim_file": sim_file, "tra_file": tra_file, "geo_file": geo_file},
    )
    print(f"Saved trained model to {Path(out_file).resolve()}")

    if use_wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: the geo/sim/tra paths, output path, and run options.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the β⁺ annihilation model on a full .sim and save it for later inference."
    )
    parser.add_argument("--geo-file", required=True, help="MEGAlib geometry setup file")
    parser.add_argument("--sim-file", required=True, help="MEGAlib .sim file (simulated events) to train on")
    parser.add_argument("--tra-file", required=True, help="MEGAlib .tra file (Compton events)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pt model path (default: models/<sim-stem>.pt)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--search-iters",
        type=int,
        default=50,
        help="Champion-seeded search candidates (0 keeps the champion seed; cost scales with this x folds)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the decision-threshold calibration (keep the raw count-prior operating point)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = args.out or str(MODELS_DIR / f"{Path(args.sim_file).stem}.pt")
    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
    train_and_save(
        args.geo_file,
        args.sim_file,
        args.tra_file,
        out_path,
        args.wandb,
        args.search_iters,
        not args.no_calibrate,
    )
