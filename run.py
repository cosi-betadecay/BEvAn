import ROOT as M
import math
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),     
        logging.FileHandler("logs/run.log")
    ]
)

def evaluate_annihilation_detector(geometry_file: str, sim_file: str,
                                    energy: int = 511, tolerance : float = 6.0,
                                    make_plots: bool = True) -> dict:
    """
    Evaluate a simple annihilation detector based only on HT (hit) information.

    Args:
        geometry_file (str): Path to the .geo.setup file
        sim_file (str): Path to the .sim file
        energy (float): Expected energy (default 511 keV)
        tolerance (float): Allowed deviation (default ±6 keV)
        make_plots (bool): If True, plots will be generated 
                           (histogram of energy sums + precision-recall curve)

    Returns:
        dict: Results containing TP, FP, TN, FN, precision, and recall
    """

    # Load MEGAlib
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    Geometry = M.MDGeometryQuest()
    if not Geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")

    Reader = M.MFileEventsSim(Geometry)
    if not Reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")

    TP = FP = TN = FN = 0
    all_sums = []

    while True:
        Event = Reader.GetNextEvent()
        if not Event:
            break
        M.SetOwnership(Event, True)

        is_true_anni = any(
            Event.GetIAAt(i).GetProcess() == M.MString("ANNI")
            for i in range(Event.GetNIAs())
        )

        detected = False
        n_ht = Event.GetNHTs()
        for i in range(n_ht):
            for j in range(i+1, n_ht):
                e_sum = Event.GetHTAt(i).GetEnergy() + Event.GetHTAt(j).GetEnergy()
                all_sums.append(e_sum)
                if abs(e_sum - energy) < tolerance:
                    detected = True
                    break
            if detected:
                break

        if detected and is_true_anni:
            TP += 1
        elif detected and not is_true_anni:
            FP += 1
        elif not detected and is_true_anni:
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    results = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
    }

    logging.info("=== Results ===")
    for k, v in results.items():
        logging.info(f"{k}: {v}")

    if make_plots:
        plt.figure(figsize=(8,4))
        plt.hist(all_sums, bins=100, range=(0,1000), histtype='step', color='blue')
        plt.axvline(energy, color='red', linestyle='--', label=f"{energy} keV")
        plt.xlabel("Energy sum [keV]")
        plt.ylabel("Number of pairs")
        plt.title("HT pair energy sums")
        plt.legend()
        plt.savefig("plots/ht_pair_e_sum.png")

        precisions, recalls, tolerances = [], [], list(range(2,20,2))
        for tol in tolerances:
            res = evaluate_annihilation_detector(geometry_file, sim_file, energy, tol, make_plots=False)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
        
        plt.figure(figsize=(6,6))
        plt.plot(recalls, precisions, marker='o')
        for i, tol in enumerate(tolerances):
            plt.text(recalls[i], precisions[i], str(tol))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve vs tolerance")
        plt.grid(True)
        plt.savefig("plots/precision_recall.png")

    return results

