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

from mathematics.calculations import calculate_tolerance, min_max_norm
from physics.event_processing import event_data_processing
from physics.ground_truths import (
    ground_truth_annihilation,
    ground_truth_bdecay,
    ground_truth_compton,
)
from physics.posterior import posterior
from utils.calculations import log_calculations
from utils.plots import (
    cdf,
    ecdf_slope,
    histogram,
    histogram_log,
    violin_plot,
)
from utils.reader_extraction import get_reader


def detected_511_event_posterior(
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
):
    energies, positions = event_data_processing(event, cfg)
    if energies is None:
        return 0.0

    _posterior, _ = posterior(
        energies,
        positions,
        511,
        cfg_likelihoods=cfg.likelihoods,
        tolerance=calculate_tolerance(),
    )

    return _posterior.max()


def posteriorhilation_extractor_posterior(
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

        score_posterior = detected_511_event_posterior(event, cfg)
        _ground_truth_bdecay = ground_truth_bdecay(event, ref_energy)
        _ground_truth_annihilation = ground_truth_annihilation(event)
        _ground_truth_compton = ground_truth_compton(event)

        scores_bdecay.append(score_posterior)
        ground_truths_beta_decay.append(_ground_truth_bdecay)
        ground_truths_annihilation.append(_ground_truth_annihilation)
        ground_truths_compton.append(_ground_truth_compton)

    scores_bdecay = torch.tensor(scores_bdecay, dtype=torch.float32)
    scores_bdecay = min_max_norm(scores_bdecay, basis=scores_bdecay)
    ground_truths_beta_decay = torch.tensor(ground_truths_beta_decay, dtype=torch.bool)
    ground_truths_annihilation = torch.tensor(ground_truths_annihilation, dtype=torch.bool)
    ground_truths_compton = torch.tensor(ground_truths_compton, dtype=torch.bool)

    return scores_bdecay, ground_truths_beta_decay, ground_truths_annihilation, ground_truths_compton


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base=None,
)
def main(
    cfg: omegaconf.dictconfig.DictConfig,
) -> None:
    """Run posteriorhilation detection extraction.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
    """
    wandb.init(project="cosi-betadecay-posterior")

    scores_bdecay, ground_truths_bdecay, ground_truths_annihilation, ground_truths_compton = (
        posteriorhilation_extractor_posterior(cfg)
    )

    # Beta decay
    true_events_posterior_bdecay = []
    false_events_posterior_bdecay = []
    # posteriorhilations
    true_events_posterior_annihilation = []
    false_events_posterior_annihilation = []
    # Compton
    true_events_posterior_compton = []
    false_events_posterior_compton = []

    # Mapping runs
    for i in tqdm(range(len(ground_truths_bdecay)), desc="Creating results", unit=" events"):
        if ground_truths_bdecay[i]:
            true_events_posterior_bdecay.append(scores_bdecay[i].item())
        else:
            false_events_posterior_bdecay.append(scores_bdecay[i].item())
        if ground_truths_annihilation[i]:
            true_events_posterior_annihilation.append(scores_bdecay[i].item())
        else:
            false_events_posterior_annihilation.append(scores_bdecay[i].item())
        if ground_truths_compton[i]:
            true_events_posterior_compton.append(scores_bdecay[i].item())
        else:
            false_events_posterior_compton.append(scores_bdecay[i].item())

    # Creating runs
    runs = [
        ("posterior_bdecay", scores_bdecay.cpu().numpy()),
        ("posterior_true_events_bdecay", true_events_posterior_bdecay),
        ("posterior_false_events_bdecay", false_events_posterior_bdecay),
        ("posterior_annihilation", scores_bdecay.cpu().numpy()),
        ("posterior_true_events_annihilation", true_events_posterior_annihilation),
        ("posterior_false_events_annihilation", false_events_posterior_annihilation),
        ("posterior_compton", scores_bdecay.cpu().numpy()),
        ("posterior_true_events_compton", true_events_posterior_compton),
        ("posterior_false_events_compton", false_events_posterior_compton),
    ]

    load_dotenv()

    for label, data in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-posterior", name=label, reinit=True)

        histogram(data, label)
        histogram_log(data, label)
        cdf(data, label)
        violin_plot(data, label)
        ecdf_slope(data, label)
        log_calculations(data, label)

    wandb.finish()


if __name__ == "__main__":
    main()
