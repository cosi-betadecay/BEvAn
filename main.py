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
    cfg: omegaconf.dictconfig.DictConfig,
) -> None:
    """Run annihilation detection extraction.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
    """
    wandb.init(project=cfg.project_names[0])

    annihilation_extractor(cfg)

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    main()
