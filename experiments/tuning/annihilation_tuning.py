import os
import sys

import hydra
import omegaconf
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

    true_events_indicies = annihilation_extractor(cfg)

    wandb.finish()


if __name__ == "__main__":
    main()
