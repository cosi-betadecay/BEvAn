import matplotlib.pyplot as plt
import omegaconf
import ROOT as M
import torch
import wandb
from event_processing import detected_511_event
from tqdm import tqdm

from mathematics.calculations import min_max_norm
from physics.bayesian_annihilation import BayesianAnnihiliationModel
from physics.ground_truths import ground_truth_bdecay
from utils.plots import plot_confusion_matrix
from utils.reader_extraction import get_reader


def annihilation_extractor(
    cfg: omegaconf.dictconfig.DictConfig,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    ground_truths = []
    scores_bdecay = []
    scores_bg = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)

        score_bdecay, score_bg = detected_511_event(ref_energy, event, cfg)
        _ground_truth = ground_truth_bdecay(event, ref_energy)

        scores_bdecay.append(score_bdecay)
        scores_bg.append(score_bg)
        ground_truths.append(_ground_truth)

    scores_bdecay = torch.tensor(scores_bdecay, dtype=torch.float32)
    scores_bg = torch.tensor(scores_bg, dtype=torch.float32)
    ground_truths = torch.tensor(ground_truths, dtype=torch.bool)
    joined_scores = torch.stack([scores_bdecay, scores_bg], dim=1)

    scores_bdecay = min_max_norm(scores_bdecay, basis=joined_scores)
    scores_bg = min_max_norm(scores_bg, basis=joined_scores)

    predictions = BayesianAnnihiliationModel(scores_bdecay, scores_bg, 1, 1).inference()

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
