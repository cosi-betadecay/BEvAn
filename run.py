import ROOT as M
import matplotlib.pyplot as plt
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),     
        logging.FileHandler("logs/run.log")
    ]
)

def get_reader(geometry_file: str, sim_file: str):
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    Geometry = M.MDGeometryQuest()
    if not Geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")
    else:
        logging.info(f"Geometry {geometry_file} loaded!")

    Reader = M.MFileEventsSim(Geometry)
    if not Reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")
    else:
        logging.info(f"Simulation file {sim_file} opened!")
        
    return Reader


def process(Event, ref_energy=511, tolerance=5.0):
    NumberGoodEvents = 0

    for i in range(Event.GetNIAs()):
        if Event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            ProcessID = i + 1

            SecondaryIDs = []
            for j in range(Event.GetNIAs()):
                if Event.GetIAAt(j).GetOriginID() == ProcessID:
                    SecondaryIDs.append(j + 1)

            TotalEnergy = 0
            for h in range(Event.GetNHTs()):
                for SID in SecondaryIDs:
                    if Event.GetHTAt(h).IsOrigin(SID):
                        TotalEnergy += Event.GetHTAt(h).GetEnergy()
                        break

            if abs(TotalEnergy - ref_energy) < tolerance:
                NumberGoodEvents += 1

    return NumberGoodEvents



def detected_511_event(ref_energy, tolerance, Event):
    energy_sums = {}

    n_hits = Event.GetNHTs()
    for i in range(n_hits):
        energy = Event.GetHTAt(i).GetEnergy()

        for j in range(Event.GetHTAt(i).GetNOrigins()):
            origin_id = Event.GetHTAt(i).GetOriginAt(j)
            if origin_id not in energy_sums:
                energy_sums[origin_id] = 0.0
            energy_sums[origin_id] += energy

    for total in energy_sums.values():
        if abs(total - ref_energy) < tolerance:
            return True

    return False

def annihilation_extractor_v2(geometry_file: str, sim_file: str,
                            ref_energy: int = 511, tolerance: float = 6.0):

    Reader = get_reader(geometry_file, sim_file)

    TP, FP, FN, TN = 0, 0, 0, 0

    while True:
        Event = Reader.GetNextEvent()
        if not Event:
            break
        M.SetOwnership(Event, True)

        NumberGoodEvents = process(Event, ref_energy, tolerance)
        is_annihilation = (NumberGoodEvents > 0)
        detected_511 = detected_511_event(ref_energy, tolerance, Event)

        if is_annihilation:
            if detected_511:
                TP += 1
            else:
                FN += 1
        else:
            if detected_511:
                FP += 1
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0

    logging.info(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    logging.info(f"Precision: {precision:.3f}")
    logging.info(f"Recall: {recall:.3f}")
    logging.info(f"False Positive Rate: {fpr:.3f}")

    return TP, FP, FN, TN, precision, recall, fpr


""""
Plan:
1:
Start from each annihilation-photon (IA in MEGAlib).
Follow whole Compton-chain: all HTs that shares same OriginID. (Think of a three from algorithms and data structures?)
Sum all energies for this chain → we get estimated gamma-energy.

2:
Create a probability distribution from these results
Classify based on the results of the distribtion
Start with Gaussian, then go over to more "complex" if needed

3:
Create some plots of the results, etc.
"""