import torch

from physics.preprocessing.compton_kin import compton_kin_preprocessing
from physics.preprocessing.energy import energy_window
from physics.preprocessing.radial_ordering import radial_ordering_preprocessing


def preprocesser(energies: torch.Tensor, positions: torch.Tensor, sigma: float) -> bool:
    new_energies_v1, new_positions_v1 = energy_window(energies, positions, sigma)
    new_energies_v2, new_positions_v2 = compton_kin_preprocessing(new_energies_v1, new_positions_v1)
    new_energies_v3, new_positions_v3 = radial_ordering_preprocessing(new_energies_v2, new_positions_v2)

    return new_energies_v3, new_positions_v3
