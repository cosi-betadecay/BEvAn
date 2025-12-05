import os

import numpy as np
import wandb
from dotenv import load_dotenv

from physics.annihilation_detection import annihilation_extractor
from utils.plots import plot_metrics

def simulation() -> None:
    """Run the annihilation event simulation and evaluate detection performance."""
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    wandb.init(project="cosi-betadecay")

    tolerances = np.array([10, 9, 8, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1])

    f1_scores = np.zeros_like(tolerances, dtype=float)
    precision_scores = np.zeros_like(tolerances, dtype=float)
    recall_scores = np.zeros_like(tolerances, dtype=float)
    fpr_scores = np.zeros_like(tolerances, dtype=float)

    for idx, tol in enumerate(tolerances):
        (TP, FP, FN, TN, precision, recall, fpr, f1) = annihilation_extractor(
            "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
            "Activation.sim",
            tolerance=tol,
        )

        precision_scores[idx] = precision
        recall_scores[idx] = recall
        fpr_scores[idx] = fpr
        f1_scores[idx] = f1

    plot_metrics(tolerances, precision_scores, recall_scores, fpr_scores, f1_scores)

    wandb.finish()


if __name__ == "__main__":
    simulation()
