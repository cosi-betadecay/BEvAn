import argparse
import os
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv

from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.datasets import Datasets
from pipeline.priors import (
    apply_prior_counts,
    count_prior_events,
    validate_prior_counts,
    validate_prior_sims,
)
from pipeline.train import Trainer, flatten_config
from utils.reader_extraction import load_geometry, open_sim_reader
from utils.wandb_logging import log_density_terms

PROJECT = "cosi-betadecay"
MODELS_DIR = Path("models")


def train_and_save(
    geo_file: str,
    sim_file: str,
    tra_file: str,
    out_file: str,
    use_wandb: bool,
    prior_sims: list[str],
) -> None:
    """Fit the per-bucket model on a whole simulation and save it to a torch artifact.

    Unlike :mod:`analysis`, this fits on every event in the simulation (no
    train/eval split): held-out testing happens later on a separate simulation
    via :mod:`inference`. The fixed reference config is identical to the
    ``analysis.py`` selection, so the saved model's operating point matches what
    ``analysis.py`` would deploy. The fitted densities — with the measured prior
    counts baked in — are written to ``out_file`` for later inference.

    Args:
        geo_file: MEGAlib geometry setup file.
        sim_file: MEGAlib ``.sim`` file (simulated events) to train on.
        tra_file: MEGAlib ``.tra`` file (Compton events) for source reconstruction.
        out_file: Destination path for the saved ``.pt`` model artifact.
        use_wandb: Whether to log selection diagnostics to Weights & Biases.
        prior_sims: Dedicated prior ``.sim`` files the per-bucket class priors
            are counted from (see :mod:`pipeline.priors`); the training data is
            never used for the prior.

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
    validate_prior_sims(prior_sims)

    # Count the class priors from the dedicated prior files before the pipeline
    # begins: a bad prior file fails fast, before the backprojection and
    # feature extraction.
    prior_counts = count_prior_events(geo_file, prior_sims)
    validate_prior_counts(prior_counts)

    if use_wandb:
        wandb.init(project=PROJECT, name=f"train:{sim_path.stem}")

    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    imager.backproject_file(tra_file)
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = open_sim_reader(load_geometry(geo_file), sim_file)

    datasets = Datasets(reader_sim, unit_vector)
    data = datasets.compute_event_features()

    # Fit on the full simulation (no train/eval split) at the fixed reference
    # config, matching the analysis.py selection: the held-out test set is a
    # separate simulation run through inference.py.
    trainer = Trainer()
    flat = flatten_config(trainer.config())
    if use_wandb:
        wandb.log(dict(flat))

    trainer.fit(data)

    # The prior counted upfront replaces the training counts.
    apply_prior_counts(trainer, prior_counts)

    # Log the fitted per-bucket density terms (model-level; no-op without a run).
    log_density_terms(trainer)

    provenance = {
        "sim_file": sim_file,
        "tra_file": tra_file,
        "geo_file": geo_file,
        "prior_sim_files": list(prior_sims),
        "prior_counts": prior_counts,
    }
    trainer.save(out_file, provenance=provenance)

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
        help="Output .pt model path (default: models/<sim-stem>_<timestamp>.pt)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--prior-sim",
        nargs="+",
        required=True,
        metavar="SIM",
        help="One or more dedicated prior .sim files; the per-bucket class priors are counted "
        "from these files (the training data is never used for the prior)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.out or str(MODELS_DIR / f"{Path(args.sim_file).stem}_{timestamp}.pt")
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
        args.prior_sim,
    )
