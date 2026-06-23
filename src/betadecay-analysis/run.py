import argparse
import os
from pathlib import Path

from dataset.datasets import Datasets
from dotenv import load_dotenv
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

import wandb

PROJECT = "cosi-betadecay"


def main(geo_file: str, sim_file: str, tra_file: str, use_wandb: bool) -> None:
    """Run the β⁺ annihilation detection pipeline on a given MEGAlib dataset.

    Args:
        geo_file (str): Path to the MEGAlib geometry setup file.
        sim_file (str): Path to the MEGAlib .sim file containing simulated events.
        tra_file (str): Path to the MEGAlib .tra file containing Compton events.
        use_wandb (bool): Whether to use Weights & Biases for logging.

    Raises:
        ValueError: If any of the input files have the wrong extension.
        FileNotFoundError: If any of the input files do not exist.
        FileNotFoundError: If any of the input files do not exist.
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

    trainer = Trainer().fit(train)
    Evaluator(trainer).evaluate(eval_data, split_name="eval")

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
        default="$(MEGALIB)/resource/examples/geomega/special/SPILike.geo.setup",
        help="MEGAlib geometry setup file",
    )
    parser.add_argument("--sim-file", default="data/SPILike.sim", help="MEGAlib .sim file (simulated events)")
    parser.add_argument("--tra-file", default="data/SPILike.tra", help="MEGAlib .tra file (Compton events)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
    main(args.geo_file, args.sim_file, args.tra_file, args.wandb)
