import itertools
from typing import Any

import matplotlib.pyplot as plt
import ROOT as M
import torch
from tqdm import tqdm

import wandb
from mathematics.calculations import calculate_tolerance
from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_pdf_bdecay
from physics.likelihoods.kn import klein_nishina_pdf
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
    """_summary_

    Args:
        posterior_bdecay (float): _description_
        posterior_bg (float): _description_

    Returns:
        bool: _description_
    """
    return posterior_bdecay >= 0.85


def detected_511_event(
    ref_energy: float,
    event: Any,
    alpha_energy: float,
    alpha_arm: float,
) -> bool:
    """Determine if an event is detected as a 511 keV annihilation event based on posterior probability.

    Args:
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        event (Any): MEGAlib event object containing interactions and hits.
        alpha_compton_kin (float): Compton kinetic energy parameter.
        alpha_kn (float): Klein-Nishina parameter.
        alpha_mid (float): Maximum interaction distance parameter.
        alpha_arm (float): Angular resolution measure parameter.
    Returns:
        bool: True if the event is detected as a 511 keV annihilation event, else False.
    """
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()

    energies = torch.tensor([event.GetHTAt(i).GetEnergy() for i in range(n_hits)], dtype=torch.float32)
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
    )

    _best = -float("inf")
    _kn = -float("inf")

    one = torch.tensor(1.0, dtype=energies.dtype, device=energies.device)

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]


            _arm = angular_resolution_measure_kernel(energy_combo, pos_combo) if r > 2 else one
            _kn = klein_nishina_pdf(energy_combo) if r > 1 else one
            _energy = energy_pdf_bdecay(energy_combo, ref_energy, tolerance)

            _arm = torch.clamp(_arm, min=1e-12)
            _kn = torch.clamp(_kn, min=1e-12)
            _energy = torch.clamp(_energy, min=1e-12)

            alpha_arm = 1
            alpha_energy = 1
            alpha_kn = 0

            score = (
                alpha_arm * torch.log(_arm)
                + alpha_energy * (_energy)
                + alpha_kn * torch.log(_kn)
            )

            _best = max(_best, score.item())
    
    #print(score)

    return classifier(_best, 0), _kn


def annihilation_extractor(
    geometry_file: str,
    sim_file: str,
    alpha_energy: float,
    alpha_arm: float,
    ref_energy: int = 511,
) -> None:
    """Extract annihilation events and compute detection performance metrics.

    Args:
        geometry_file (str): Path to the detector geometry setup file (XML).
        sim_file (str): Path to the MEGAlib simulation file (.sim).
        alpha_compton_kin (float): Compton kinetic energy parameter.
        alpha_kn (float): Klein-Nishina parameter.
        alpha_mid (float): Maximum interaction distance parameter.
        alpha_arm (float): Angular resolution measure parameter.
        ref_energy (float): Reference energy to compare against (default 511 keV).
    """
    reader = get_reader(geometry_file, sim_file)

    ground_truths = []
    predictions = []
    kns = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        prediction, _kn = detected_511_event(ref_energy, event, alpha_energy, alpha_arm)
        _ground_truth = ground_truth(event, ref_energy)

        predictions.append(prediction)
        kns.append(_kn)
        ground_truths.append(_ground_truth)

    kns = torch.tensor(kns)
    predictions = torch.tensor(predictions, dtype=torch.bool)
    gt = torch.tensor(ground_truths, dtype=torch.bool)

    # Apply KN cut and gate predictions
    quantile = torch.quantile(kns, 0.05)
    kns_pred = kns >= quantile
    preds = predictions

    tp = torch.sum(gt & preds).item()
    fp = torch.sum(~gt & preds).item()
    fn = torch.sum(gt & ~preds).item()
    tn = torch.sum(~gt & ~preds).item()

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
