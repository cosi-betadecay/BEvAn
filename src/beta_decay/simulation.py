import os

import wandb
from dotenv import load_dotenv
from physics.annihilation_detection import annihilation_extractor


def simulation(
    run_name: str,
    baseline_activation: bool,
    compton_angle_activation: bool,
    mid_activation: bool,
    arm_activation: bool,
) -> None:
    """Run the annihilation event simulation and evaluate detection performance."""
    wandb.init(
        project="cosi-betadecay",
        name=run_name,
        config={
            "baseline_activation": baseline_activation,
            "compton_angle_activation": compton_angle_activation,
            "mid_activation": mid_activation,
            "arm_activation": arm_activation,
        },
    )

    annihilation_extractor(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
        "data/Activation.sim",
        baseline_activation,
        compton_angle_activation,
        mid_activation,
        arm_activation,
    )

    wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    simulation_config = {
        "baseline": [True, False, False, False],
        "baseline_compton-kin": [True, True, False, False],
        "baseline_mid": [True, False, True, False],
        "baseline_arm": [True, False, False, True],
        "baseline_compton-kin_mid": [True, True, True, False],
        "baseline_compton-kin_arm": [True, True, False, True],
        "baseline_mid_arm": [True, False, True, True],
        "baseline_compton-kin_mid_arm": [True, True, True, True],
        "compton-kin": [False, True, False, False],
        "compton-kin_mid": [False, True, True, False],
        "compton-kin_arm": [False, True, False, True],
        "compton-kin_mid_arm": [False, True, True, True],
        "mid": [False, False, True, False],
        "mid_arm": [False, False, True, True],
        "arm": [False, False, False, True],
        "no_filters": [False, False, False, False],
    }
    for run_name, filters in simulation_config.items():
        simulation(run_name, *filters)
