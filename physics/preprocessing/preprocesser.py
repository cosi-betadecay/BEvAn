import torch

from physics.preprocessing.compton_kin import compton_kin
from physics.preprocessing.energy import energy_window


def preprocesser(energies: torch.Tensor, positions: torch.Tensor, sigma: float) -> bool:
    return compton_kin(energies) and energy_window(energies, sigma)
