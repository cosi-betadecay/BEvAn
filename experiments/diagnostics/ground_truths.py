import os
import sys
from typing import Any

import hydra
import omegaconf
import ROOT as M
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.reader_extraction import get_reader


def ground_truth_annihilation(event: Any) -> bool:
    """Determine if an event contains an annihilation interaction with energy near 511 keV.

    Args:
        event (Any): MEGAlib event object containing interactions and hits.
        ref_energy (float): Reference energy for annihilation events (e.g., 511 keV).

    Returns:
        bool: True if an annihilation interaction is present with energy near ref_energy, else False.
    """
    for i in range(event.GetNIAs()):
        print(event.GetIAAt(i))
        # if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
        #    return True

    # return False


def ground_truth_compton(event: Any) -> bool:
    """Determine if an event contains a Compton scattering interaction.

    Args:
        event (Any): MEGAlib event object containing interactions and hits.

    Returns:
        bool: True if a Compton scattering interaction is present, else False.
    """
    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("COMP"):
            return True

    return False


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    reader = get_reader(cfg.setup.geo_file, cfg.setup.sim_file)

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)
        ground_truth_annihilation(event)


if __name__ == "__main__":
    main()
