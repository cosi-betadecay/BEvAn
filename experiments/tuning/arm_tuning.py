import os
import sys
from typing import Any

import hydra
import omegaconf
import ROOT as M
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mathematics.calculations import min_max_norm
from physics.event_processing import event_data_processing
from physics.ground_truths import (
    ground_truth_annihilation,
    ground_truth_bdecay,
    ground_truth_compton,
)
from physics.likelihoods.arm import angular_resolution_measure_kernel
from utils.calculations import log_calculations
from utils.plots import (
    cdf,
    ecdf_slope,
    histogram,
    histogram_log,
    violin_plot,
)
from utils.reader_extraction import get_reader

##################################################################################


def detected_511_event_arm(
    ref_energy: float,
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energies, positions = event_data_processing(event, cfg)
    if energies is None:
        return 0.0
    n_combo = energies.shape[0]

    sizes = (energies > 0).sum(dim=1)
    one = torch.ones(n_combo, device=device)

    arm = one.clone()
    valid_arm = sizes > 2
    if valid_arm.any():
        arm[valid_arm] = angular_resolution_measure_kernel(
            energies[valid_arm],
            positions[valid_arm],
            cfg.likelihoods,
        )

    return arm.max()


def annihilation_extractor_arm(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    scores_arm = []
    ground_truths_beta_decay = []
    ground_truths_annihilation = []
    ground_truths_compton = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        score_arm = detected_511_event_arm(ref_energy, event, cfg)
        _ground_truth_bdecay = ground_truth_bdecay(event, ref_energy)
        _ground_truth_anni = ground_truth_annihilation(event)
        _ground_truth_compton = ground_truth_compton(event)

        scores_arm.append(score_arm)
        ground_truths_beta_decay.append(_ground_truth_bdecay)
        ground_truths_annihilation.append(_ground_truth_anni)
        ground_truths_compton.append(_ground_truth_compton)

    scores_arm = torch.tensor(scores_arm, dtype=torch.float32)
    scores_arm = min_max_norm(scores_arm, basis=scores_arm)
    ground_truths_beta_decay = torch.tensor(ground_truths_beta_decay, dtype=torch.bool)
    ground_truths_annihilation = torch.tensor(ground_truths_annihilation, dtype=torch.bool)
    ground_truths_compton = torch.tensor(ground_truths_compton, dtype=torch.bool)

    return scores_arm, ground_truths_beta_decay, ground_truths_annihilation, ground_truths_compton


@hydra.main(
    config_path="../../configs",
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
    scores_arm, ground_truths_bdecay, ground_truths_annihilation, ground_truths_compton = (
        annihilation_extractor_arm(cfg)
    )

    # Beta decay
    true_events_arm_bdecay = []
    false_events_arm_bdecay = []
    # Annihilations
    true_events_arm_annihilation = []
    false_events_arm_annihilation = []
    # Compton
    true_events_arm_compton = []
    false_events_arm_compton = []

    # Mapping runs
    for i in tqdm(range(len(ground_truths_bdecay)), desc="Creating results", unit=" events"):
        if ground_truths_bdecay[i]:
            true_events_arm_bdecay.append(scores_arm[i].item())
        else:
            false_events_arm_bdecay.append(scores_arm[i].item())
        if ground_truths_annihilation[i]:
            true_events_arm_annihilation.append(scores_arm[i].item())
        else:
            false_events_arm_annihilation.append(scores_arm[i].item())
        if ground_truths_compton[i]:
            true_events_arm_compton.append(scores_arm[i].item())
        else:
            false_events_arm_compton.append(scores_arm[i].item())

    # Creating runs
    runs = [
        ("arm_bdecay", scores_arm.cpu().numpy()),
        ("arm_true_events_bdecay", true_events_arm_bdecay),
        ("arm_false_events_bdecay", false_events_arm_bdecay),
        ("arm_annihilation", scores_arm.cpu().numpy()),
        ("arm_true_events_annihilation", true_events_arm_annihilation),
        ("arm_false_events_annihilation", false_events_arm_annihilation),
        ("arm_compton", scores_arm.cpu().numpy()),
        ("arm_true_events_compton", true_events_arm_compton),
        ("arm_false_events_compton", false_events_arm_compton),
    ]

    load_dotenv()

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-arm", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

    wandb.finish()


if __name__ == "__main__":
    main()
