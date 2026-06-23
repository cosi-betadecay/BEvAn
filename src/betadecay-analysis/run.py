import argparse
import os

from dataset.datasets import Datasets
from dotenv import load_dotenv
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

import wandb

PROJECT = "cosi-betadecay"


def main(geo_file: str, sim_file: str, tra_file: str, use_wandb: bool) -> None:
    """Run β⁺ annihilation detection on a single (geo, sim, tra) dataset."""
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
