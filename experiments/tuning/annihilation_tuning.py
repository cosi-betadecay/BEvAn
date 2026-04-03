import os
import sys
from typing import Any

import hydra
import omegaconf
import ROOT as M
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mathematics.calculations import calculate_tolerance, min_max_norm
from physics.event_processing import event_data_processing
from physics.ground_truths import (
    ground_truth_annihilation,
    ground_truth_bdecay,
    ground_truth_compton,
)
from physics.likelihoods.annihilation import annihilation_kernel
from physics.likelihoods.energy import energy_kernel_bdecay
from utils.calculations import log_calculations
from utils.plots import cdf, ecdf_slope, histogram, histogram_log, violin_plot
from utils.reader_extraction import get_reader


def detected_511_event_anni(
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
    tolerance = calculate_tolerance()

    anni = one.clone()
    valid_anni = sizes > 3
    if valid_anni.any():
        anni[valid_anni] = annihilation_kernel(positions[valid_anni])

    eng = energy_kernel_bdecay(energies, 511, tolerance)

    score = eng * anni

    return score.max()


def annihilation_extractor_anni(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    ground_truths_beta_decay = []
    ground_truths_annihilation = []
    ground_truths_compton = []
    scores_bdecay = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        score_anni = detected_511_event_anni(event, cfg)
        _ground_truth_bdecay = ground_truth_bdecay(event, ref_energy)
        _ground_truth_anni = ground_truth_annihilation(event)
        _ground_truth_compton = ground_truth_compton(event)

        scores_bdecay.append(score_anni)
        ground_truths_beta_decay.append(_ground_truth_bdecay)
        ground_truths_annihilation.append(_ground_truth_anni)
        ground_truths_compton.append(_ground_truth_compton)

    scores_bdecay = torch.tensor(scores_bdecay, dtype=torch.float32)
    scores_bdecay = min_max_norm(scores_bdecay, basis=scores_bdecay)
    ground_truths_beta_decay = torch.tensor(ground_truths_beta_decay, dtype=torch.bool)

    return scores_bdecay, ground_truths_beta_decay, ground_truths_annihilation, ground_truths_compton


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
    wandb.init(project="cosi-betadecay-annihilation")

    scores_bdecay, ground_truths_bdecay, ground_truths_annihilation, ground_truths_compton = (
        annihilation_extractor_anni(cfg)
    )

    # Beta decay
    true_events_anni_bdecay = []
    false_events_anni_bdecay = []
    # Annihilations
    true_events_anni_annihilation = []
    false_events_anni_annihilation = []
    # Compton
    true_events_anni_compton = []
    false_events_anni_compton = []

    # Mapping runs
    for i in tqdm(range(len(ground_truths_bdecay)), desc="Creating results", unit=" events"):
        if ground_truths_bdecay[i]:
            true_events_anni_bdecay.append(scores_bdecay[i].item())
        else:
            false_events_anni_bdecay.append(scores_bdecay[i].item())
        if ground_truths_annihilation[i]:
            true_events_anni_annihilation.append(scores_bdecay[i].item())
        else:
            false_events_anni_annihilation.append(scores_bdecay[i].item())
        if ground_truths_compton[i]:
            true_events_anni_compton.append(scores_bdecay[i].item())
        else:
            false_events_anni_compton.append(scores_bdecay[i].item())

    # Creating runs
    runs = [
        ("anni_bdecay", scores_bdecay.cpu().numpy()),
        ("anni_true_events_bdecay", true_events_anni_bdecay),
        ("anni_false_events_bdecay", false_events_anni_bdecay),
        ("anni_annihilation", scores_bdecay.cpu().numpy()),
        ("anni_true_events_annihilation", true_events_anni_annihilation),
        ("anni_false_events_annihilation", false_events_anni_annihilation),
        ("anni_compton", scores_bdecay.cpu().numpy()),
        ("anni_true_events_compton", true_events_anni_compton),
        ("anni_false_events_compton", false_events_anni_compton),
    ]

    load_dotenv()

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-energy", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

    wandb.finish()


if __name__ == "__main__":
    main()
