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
    cm = np.array([[TP, FN], [FP, TN]])
    labels = ["Positive", "Negative"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.show()
    wandb.log({"confusion_matrix": wandb.Image(plt)})
