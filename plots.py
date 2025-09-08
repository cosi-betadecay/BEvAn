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