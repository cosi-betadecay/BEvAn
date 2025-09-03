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


def annihilation_extractor(geometry_file: str, sim_file: str,
                            energy: int = 511, tolerance : float = 5):
    
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    Geometry = M.MDGeometryQuest()
    if not Geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")

    Reader = M.MFileEventsSim(Geometry)
    if not Reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")

    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(geometry_file)):
        logging.info("Geometry " + geometry_file + " loaded!")
    else:
        logging.info("Unable to load geometry " + geometry_file + " - Aborting!")
        quit()
        

    Reader = M.MFileEventsSim(Geometry)
    if not Reader.Open(M.MString(sim_file)):
        logging.info("Unable to open file " + sim_file + ". Aborting!")
        quit()

    NumberGoodEvents = 0
    NumberBackgroundEvents = 0
    TP, FP, FN, TN = 0, 0, 0, 0

    while True: 
        Event = Reader.GetNextEvent()
        if not Event:
            break
        M.SetOwnership(Event, True)
        
        NumberANNI = 0
        for i in range(0, Event.GetNIAs()):
            if Event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
                NumberANNI += 1
                ProcessID = i+1
                SecondaryIDs = []
                for i in range(0, Event.GetNIAs()):
                    if Event.GetIAAt(i).GetOriginID() == ProcessID:
                        SecondaryIDs.append(i+1)
                TotalEnergy = 0
                for h in range(0, Event.GetNHTs()):
                    for SID in SecondaryIDs:
                        if Event.GetHTAt(h).IsOrigin(SID):
                            TotalEnergy += Event.GetHTAt(h).GetEnergy()
                            break
                if math.fabs(TotalEnergy - energy) < tolerance:
                    NumberGoodEvents += 1

        is_annihilation = (NumberANNI > 0)

        if NumberANNI == 0:
            NumberBackgroundEvents += 1
        
        detected_511 = False
        TotalEnergy = 0.0
        for h in range(0, Event.GetNHTs()):
            TotalEnergy += Event.GetHTAt(h).GetEnergy()

        if abs(TotalEnergy - energy) < tolerance:
            detected_511 = True
        
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

    logging.info(f"Number good 511 events: {NumberGoodEvents}")
    logging.info(f"Number background events: {NumberBackgroundEvents}")


###############################################################################

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



def annihilation_extractor_v2(geometry_file: str, sim_file: str,
                            energy: int = 511, tolerance: float = 6.0):

    Reader = get_reader(geometry_file, sim_file)

    TP, FP, FN, TN = 0, 0, 0, 0

    while True:
        Event = Reader.GetNextEvent()
        if not Event:
            break
        M.SetOwnership(Event, True)

        # Find something better..
        NumberANNI = 0
        for i in range(0, Event.GetNIAs()):
            if Event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
                NumberANNI += 1
                logging.info(NumberANNI)
        is_annihilation = (NumberANNI > 0)

        # Find smarter method this sucks
        detected_511 = False
        n_hits = Event.GetNHTs()
        for i in range(n_hits):
            e1 = Event.GetHTAt(i).GetEnergy()
            for j in range(i+1, n_hits):
                e2 = Event.GetHTAt(j).GetEnergy()
                if abs((e1 + e2) - 2*energy) < tolerance:
                    detected_511 = True
                    break
            if detected_511:
                break

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
Follow whole Compton-chain: all HTs that shares same OriginID.
Sum all energies for this chain → we get estimated gamma-energy.

2:
Create a probability distribution from these results
Classify based on the results of the distribtion
Start with Gaussian, then go over to more "complex" if needed

3:
Create some plots of the results, etc.
"""