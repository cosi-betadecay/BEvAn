import os

import wandb
from dotenv import load_dotenv

from physics.annihilation_detection import annihilation_extractor


def main(
    run_name: str,
    alpha_energy: float,
    alpha_compton_kin: float,
    alpha_kn: float,
    alpha_mid: float,
    alpha_arm: float,
) -> None:
    """Run the annihilation event simulation and evaluate detection performance.

    Args:
        run_name (str): Name of the wandb run.
    """
    wandb.init(project="cosi-betadecay", name=run_name)

    annihilation_extractor(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
        "data/Activation.sim",
        alpha_energy,
        alpha_compton_kin,
        alpha_kn,
        alpha_mid,
        alpha_arm,
    )

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    main("Likelihood-80p-including:posterior:energy-full_kn-full", 1, 0, 1, 0, 0)
    main("Likelihood-80p-including:posterior:energy-full_kn-half", 1, 0, 0.5, 0, 0)
    main("Likelihood-80p-including:posterior:energy-full_kn-quarter", 1, 0.25, 0, 0, 0)
