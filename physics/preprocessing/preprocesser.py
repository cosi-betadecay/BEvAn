import torch

from physics.preprocessing.energy import energy_window


def preprocesser(energies: torch.Tensor, positions: torch.Tensor, sigma: float) -> bool:
    new_energies, new_positions = energy_window(energies, positions, sigma)

    return new_energies, new_positions