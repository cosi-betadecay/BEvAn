import os

import wandb
from dotenv import load_dotenv
from physics.annihilation_detection import annihilation_extractor


def simulation() -> None:
    """Run the annihilation event simulation and evaluate detection performance."""
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    wandb.init(project="cosi-betadecay")

    annihilation_extractor(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
        "data/Activation.sim",
    )

    wandb.finish()


if __name__ == "__main__":
    simulation()
