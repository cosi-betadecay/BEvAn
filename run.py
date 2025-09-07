import ROOT as M
import logging
import itertools
from utils import get_reader
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),     
        logging.FileHandler("logs/run.log")
    ]
)

def process(Event, ref_energy, tolerance):
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
 

def detected_511_event(ref_energy, Event, tolerance):
    n_hits = Event.GetNHTs()

    energies = [Event.GetHTAt(i).GetEnergy() for i in range(n_hits)]

    if energies == []:
        return False

    positions = [(Event.GetHTAt(i).GetPosition().X(),
                  Event.GetHTAt(i).GetPosition().Y(),
                  Event.GetHTAt(i).GetPosition().Z())
                 for i in range(n_hits)]
        
    for r in range(1, n_hits+1):
        for combo in itertools.combinations(energies, r):
            if abs(sum(combo) - ref_energy) < tolerance:
                return True

    logging.info(energies)

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
        detected_511 = detected_511_event(ref_energy, Event, tolerance)

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