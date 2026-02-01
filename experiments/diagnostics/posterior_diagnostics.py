import itertools
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

from mathematics.calculations import calculate_tolerance
from physics.annihilation_detection import ground_truth
from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_kernel_bdecay
from physics.preprocessing.preprocesser import preprocesser
from utils.plots import plot_energy_vs_arm
from utils.reader_extraction import get_reader


def arm(
    energies: torch.Tensor, positions: torch.Tensor, cfg_likelihoods: omegaconf.dictconfig.DictConfig
) -> torch.Tensor:
    def theta_geo(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions if positions.ndim == 3 else positions.unsqueeze(0)
        x0 = pos[:, 0, :]  # (B, 3)
        x1 = pos[:, 1, :]  # (B, 3)
        x2 = pos[:, 2, :]  # (B, 3)
        x0, x1, x2 = x0[0], x1[0], x2[0]

        v_in = x1 - x0
        v_out = x2 - x1

        L_in = torch.clamp(torch.norm(v_in), min=1e-3)
        L_out = torch.clamp(torch.norm(v_out), min=1e-3)

        cos_theta_geo = torch.dot(v_in, v_out) / (L_in * L_out)
        cos_theta_geo = torch.clamp(cos_theta_geo, -1.0, 1.0)
        theta_geo = torch.arccos(cos_theta_geo)

        return theta_geo

    def theta_kin(energies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        electron_mass_energy = 511.0  # keV
        if energies.ndim == 1:
            E_0 = energies.sum()
            E = E_0 - energies[0]
        else:
            E_0 = energies.sum(dim=1)
            E = E_0 - energies[:, 0]

        cos_theta_kin = 1 - electron_mass_energy * (1 / E - 1 / E_0)
        cos_theta_kin = torch.clamp(cos_theta_kin, -1.0, 1.0)
        theta_kin = torch.arccos(cos_theta_kin)

        return theta_kin

    _theta_geo = theta_geo(positions)
    _theta_kin = theta_kin(energies)

    arm = _theta_geo - _theta_kin

    return arm


def detected_511_event(
    ref_energy: float,
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return False, False, False

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

    new_energy_combo, new_pos_combo = preprocesser(energy_combo, pos_combo, tolerance, cfg.preprocessing)

    if new_energy_combo.numel() == 0 or new_pos_combo.numel() == 0:
        return False, False, False

    sizes = (energy_combo > 0).sum(dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one = torch.ones(n_combo, device=device)

    # ARM
    _arm = one.clone()
    valid_arm = sizes > 2
    if valid_arm.any():
        _arm[valid_arm] = angular_resolution_measure_kernel(
            energy_combo[valid_arm],
            pos_combo[valid_arm],
            cfg.likelihoods,
        )

    # Energy
    _eng = energy_kernel_bdecay(energy_combo, ref_energy, cfg.likelihoods.energy.n_std)

    # Numerical stability
    _arm = torch.clamp(_arm, min=1e-12)
    _eng = torch.clamp(_eng, min=1e-12)

    # Weights
    alpha_energy = cfg.likelihoods.posterior.alpha_energy
    alpha_arm = cfg.likelihoods.posterior.alpha_arm

    score = alpha_arm * torch.log(_arm) + alpha_energy * _eng

    arm_ = torch.zeros(n_combo, device=device, dtype=torch.float32)

    valid_arm = sizes >= 3
    if valid_arm.any():
        arm_[valid_arm] = arm(
            energy_combo[valid_arm],
            pos_combo[valid_arm],
            cfg.likelihoods,
        )

    imax = torch.argmax(score)

    best_score = score[imax]
    best_arm = arm_[imax]

    energy_sum = energy_combo.sum(dim=1)
    best_energy_sum = energy_sum[imax]

    return best_score, best_arm, best_energy_sum


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> tuple[torch.Tensor, torch.Tensor]:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    ground_truths = []
    arms = []
    energy_sums = []
    predictions = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)


        prediction, arm, energy_sum = detected_511_event(ref_energy, event, cfg)
        _ground_truth = ground_truth(event, ref_energy)

        predictions.append(prediction)
        ground_truths.append(_ground_truth)
        arms.append(arm)
        energy_sums.append(energy_sum)

    prediction_scores = torch.tensor(predictions, dtype=torch.float32)
    gt = torch.tensor(ground_truths, dtype=torch.bool)
    arms = torch.tensor(arms, dtype=torch.float32)
    energy_sums = torch.tensor(energy_sums, dtype=torch.float32)

    return prediction_scores, gt, arms, energy_sums


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    load_dotenv()
    
    prediction_scores, gt, arms, energy_sums = annihilation_extractor(cfg)

    prediction_scores_true_events = prediction_scores[gt]
    arms_true_events = arms[gt]
    energy_sums_true_events = energy_sums[gt]

    prediction_scores_false_events = prediction_scores[~gt]
    arms_false_events = arms[~gt]
    energy_sums_false_events = energy_sums[~gt]

    runs = [
        ("All Events - Prediction Scores", prediction_scores, arms, energy_sums),
        (
            "True Events - Prediction Scores",
            prediction_scores_true_events,
            arms_true_events,
            energy_sums_true_events,
        ),
        (
            "False Events - Prediction Scores",
            prediction_scores_false_events,
            arms_false_events,
            energy_sums_false_events,
        ),
    ]

    for label, preds, arms, energy_sums in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-diagnostics", name=label, reinit=True)

        plot_energy_vs_arm(preds, energy_sums, arms, label)

        wandb.finish()


if __name__ == "__main__":
    main()
