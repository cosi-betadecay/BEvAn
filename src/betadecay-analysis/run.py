import os

import hydra
import omegaconf
import wandb
from dataset.datasets import Datasets
from dotenv import load_dotenv
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from utils.reader_extraction import get_reader


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(
    cfg: omegaconf.dictconfig.DictConfig,
) -> None:
    """Run annihilation detection extraction.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
    """
    wandb.init(project=cfg.project_names[0])

    imager = FarFieldImager(
        geometry_file=cfg.setup.geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(cfg.setup.tra_file)
    print(f"Accumulated {n_events} Compton events from {cfg.setup.tra_file}")
    unit_vector = imager.peak_direction_cartesian()

    ref_energy = 511.0
    reader_sim, reader_tra = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    datasets = Datasets(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    data = datasets.compute_event_features(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    train, eval = datasets.split_dataset(data)

    trainer = Trainer(cfg).fit(train)
    Evaluator(trainer).evaluate(eval, split_name="eval")

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    main()
