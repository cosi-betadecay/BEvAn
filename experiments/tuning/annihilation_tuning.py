import itertools
import os
import sys
from typing import Any

import hydra
import omegaconf
import ROOT as M
import torch
import wandb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mathematics.calculations import min_max_norm
from physics.ground_truths import (
    ground_truth_annihilation,
    ground_truth_bdecay,
    ground_truth_compton,
)
from physics.likelihoods.annihilation import annihilation_kernel
from utils.calculations import log_calculations
from utils.plots import cdf, ecdf_slope, histogram, histogram_log, violin_plot
from utils.reader_extraction import get_reader


def detected_511_event_anni(
    event: Any,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return False

    # Load event data onto GPU
    energies = torch.tensor(
        [event.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

    positions = torch.tensor(
        [
            (
                event.GetHTAt(i).GetPosition().X(),
                event.GetHTAt(i).GetPosition().Y(),
                event.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    for r in range(1, n_hits + 1):
        for c in itertools.combinations(range(n_hits), r):
            combos.append(c)
            sizes.append(r)

    max_r = max(sizes)
    n_combo = len(combos)

    # Pad combinations with -1
    idx = torch.full((n_combo, max_r), -1, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # idx: (n_combo, max_r), padded with -1

    energy_combo = energies[idx]  # (n_combo, max_r)
    pos_combo = positions[idx]  # (n_combo, max_r, 3)

    mask = idx >= 0
    energy_combo = torch.where(mask, energy_combo, torch.zeros_like(energy_combo))
    pos_combo = torch.where(mask[..., None], pos_combo, torch.zeros_like(pos_combo))

    one = torch.ones(n_combo, device=device)
    anni = one.clone()
    valid_anni = sizes > 3
    if valid_anni.any():
        anni[valid_anni] = annihilation_kernel(pos_combo[valid_anni])

    return anni.max()


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

        score_anni = detected_511_event_anni(event)
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
    wandb.init(project=cfg.project_names[4])

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
