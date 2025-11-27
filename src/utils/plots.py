from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

def plot_confusion_matrix(TP: int, FP: int, FN: int, TN: int) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        TP (int): Number of true positive events.
        FP (int): Number of false positive events.
        FN (int): Number of false negative events.
        TN (int): Number of true negative events.
    """
    cm = np.array([[TP, FN],
                   [FP, TN]])
    labels = ["Positive", "Negative"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="crest",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.show()
    wandb.log({"confusion_matrix": wandb.Image(plt)})


def plot_metrics(tolerances, precision_scores, recall_scores, fpr_scores, f1_scores):
    """Plot and log precision, recall, FPR, and F1 across tolerances."""
    plt.figure(figsize=(10, 6))

    plt.plot(tolerances, precision_scores, marker='o', label="Precision")
    plt.plot(tolerances, recall_scores,    marker='s', label="Recall")
    plt.plot(tolerances, fpr_scores,       marker='^', label="False Positive Rate")
    plt.plot(tolerances, f1_scores,        marker='d', label="F1-score")

    plt.xlabel("Tolerance (keV)")
    plt.ylabel("Score")
    plt.title("Performance metrics vs. energy tolerance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    wandb.log({"performance_metrics": wandb.Image(plt)})
