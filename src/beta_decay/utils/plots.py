import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(TP: int, FP: int, FN: int, TN: int):
    """Plot and save a confusion matrix heatmap.

    Args:
        TP (int): Number of true positive events.
        FP (int): Number of false positive events.
        FN (int): Number of false negative events.
        TN (int): Number of true negative events.
    """
    cm = np.array([[TP, FN], [FP, TN]])
    labels = ["Positive", "Negative"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    return fig
