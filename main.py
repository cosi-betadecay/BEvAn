import os

import hydra
import omegaconf
import wandb
from dotenv import load_dotenv

from physics.annihilation_detection import annihilation_extractor


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(
    run_name: str,
    cfg: omegaconf.dictconfig.DictConfig,
) -> None:
    """Run annihilation detection extraction.

    Args:
        run_name (str): Name of the wandb run.
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
    """
    wandb.init(project=cfg.project_names[0], name=run_name)

    annihilation_extractor(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
        "data/Activation.sim",
        cfg,
    )

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    main("Likelihood-80p-including:posterior:energy-1.0")
