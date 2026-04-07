import matplotlib.pyplot as plt
import omegaconf
import ROOT as M
import torch
import wandb
from tqdm import tqdm

from mathematics.calculations import calculate_tolerance
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)
    tolerance = calculate_tolerance()

    ground_truths = []
    bdecay_matrix = []
    bg_matrix = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        energies, positions = event_data_processing(event, cfg)
        gt = ground_truth_bdecay(event, ref_energy)
        ground_truths.append(gt)
        # TODO: put the different energies and positions into the matrix calculations
        # TODO: add the scores from the matrix calculations to the matrices

    # TODO: convert the matrices to 2D tensors
    # TODO: compute R using the matrices with vectorized operations. R should be a 1D tensor.
    # TODO: compute the predictions using the Bayesian classifier by using R (batched)
    predictions = None
    ground_truths = torch.tensor(ground_truths, dtype=torch.bool)

    tp = torch.sum(ground_truths & predictions).item()
    fp = torch.sum(~ground_truths & predictions).item()
    fn = torch.sum(ground_truths & ~predictions).item()
    tn = torch.sum(~ground_truths & ~predictions).item()

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
