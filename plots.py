import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_confusion_matrix(TP: int, FP: int, FN: int, TN: int) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        TP (int): Number of true positive events.
        FP (int): Number of false positive events.
        FN (int): Number of false negative events.
        TN (int): Number of true negative events.
    """
    now = datetime.now()

    cm = np.array([[TP, FN],
                   [FP, TN]])
    labels = ["Positive", "Negative"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="crest",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(f"plots/cm/cm_{now:%Y-%m-%d %H:%M:%S}")
    plt.show()

def plot_metrics(tolerances, precision_scores, recall_scores, fpr_scores, f1_scores):
    now = datetime.now()
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
    plt.savefig(f"plots/metrics/metrics_{now:%Y-%m-%d %H:%M:%S}")
    plt.show()