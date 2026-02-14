import itertools
import logging
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import omegaconf
import ROOT as M
import torch
import wandb
from tqdm import tqdm

from mathematics.calculations import calculate_tolerance
from physics.posterior import posterior_bdecay
from physics.postprocessing.annihilation import annihilation_kernel
from physics.preprocessing.preprocesser import preprocesser
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def ground_truth(event: Any, ref_energy: float) -> bool:
    """Count annihilation events matching a reference energy within tolerance.

    Args:
        Event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        bool: True if any event matches the annihilation criteria, else False.
    """
    tolerance = calculate_tolerance()
    number_good_events = 0

    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            process_id = i + 1

            secondary_ids = []
            for j in range(event.GetNIAs()):
                if event.GetIAAt(j).GetOriginID() == process_id:
                    secondary_ids.append(j + 1)

            total_energy = 0
            for h in range(event.GetNHTs()):
                for sid in secondary_ids:
                    if event.GetHTAt(h).IsOrigin(sid):
                        total_energy += event.GetHTAt(h).GetEnergy()
                        break

            if abs(total_energy - ref_energy) < tolerance:
                number_good_events += 1

    return number_good_events > 0


def classifier(posterior_bdecay: float, posterior_bg: float) -> bool:
    return posterior_bdecay >= 0.5


def detected_511_event(
    ref_energy: float,
    event: Any,
    cfg: omegaconf.dictconfig.DictConfig,
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    if n_hits == 0:
        return False, -1

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
        return False, -1

    best_score = posterior_bdecay(
        new_energy_combo,
        new_pos_combo,
        ref_energy,
        tolerance,
        cfg.likelihoods,
        sizes,
    )

    classification = classifier(best_score, 0)
    index = event.GetID() if classification else -1

    return classification, index


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    ground_truths = []
    predictions = []
    predicted_true_indicies = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        prediction, index = detected_511_event(ref_energy, event, cfg)
        _ground_truth = ground_truth(event, ref_energy)

        predictions.append(prediction)
        ground_truths.append(_ground_truth)
        if prediction:
            predicted_true_indicies.append(index)

    predictions = torch.tensor(predictions, dtype=torch.bool)
    gt = torch.tensor(ground_truths, dtype=torch.bool)
    predicted_true_indicies = torch.tensor(predicted_true_indicies, dtype=torch.int32)

    tp = torch.sum(gt & predictions).item()
    fp = torch.sum(~gt & predictions).item()
    fn = torch.sum(gt & ~predictions).item()
    tn = torch.sum(~gt & ~predictions).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fig = plot_confusion_matrix(tp, fp, fn, tn)
    wandb.log(
        {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "f1_score": f1_score,
            "confusion_matrix": wandb.Image(fig),
        }
    )
    plt.close(fig)

    return predictions, ground_truths, predicted_true_indicies


def postprocessor(cfg: omegaconf.dictconfig.DictConfig) -> None:
    predictions, ground_truths, predicted_true_indicies = annihilation_extractor(cfg)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Done with naive Bayesian model, now over to the postprocessing stage.")

    positions_total_tensor = []
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    for event in tqdm(range(total=len(ground_truths)), desc="Extracting data from predicted true events"):
        event = reader.GetNextEvent()
        if event.GetID() in predicted_true_indicies:
            n_hits = event.GetNHTs()
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
            positions_total_tensor.append(positions)

    total_pairs = len(positions_total_tensor) * (len(positions_total_tensor) - 1) // 2
    for pos_i, pos_j in tqdm(
        combinations(positions_total_tensor, 2), total=total_pairs, desc="Pairing annihilation candidates"
    ):
        score = annihilation_kernel(pos_i, pos_j)

        print(score)
