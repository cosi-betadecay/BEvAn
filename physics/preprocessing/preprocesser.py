import omegaconf
import torch

from physics.preprocessing.compton_kin import compton_kin_preprocessing
from physics.preprocessing.energy import energy_window


def preprocesser(
    energies: torch.Tensor,
    positions: torch.Tensor,
    sigma: float,
    cfg_preprocessing: omegaconf.dictconfig.DictConfig,
) -> bool:
    new_energies_v1, new_positions_v1 = energy_window(energies, positions, sigma, cfg_preprocessing)
    new_energies_v2, new_positions_v2 = compton_kin_preprocessing(
        new_energies_v1, new_positions_v1, cfg_preprocessing
    )
    #new_energies_v3, new_positions_v3 = arm_preprocessing(
    #    new_energies_v2, new_positions_v2, cfg_preprocessing
    #)

    #new_energies_v4, new_positions_v4 = mid_preprocessing(
    #    new_energies_v2, new_positions_v2, cfg_preprocessing
    #)

    return new_energies_v2, new_positions_v2
