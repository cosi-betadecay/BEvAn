from run import annihilation_extractor_v2
from plots import plot_confusion_matrix

if __name__ == "__main__":
    (TP, FP, FN, TN,
    precision, recall, 
    fpr) = annihilation_extractor_v2("$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup",
                                   "Activation.sim")
    plot_confusion_matrix(TP, FP, FN, TN)