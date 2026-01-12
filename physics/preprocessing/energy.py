import torch


def energy_window(energies: torch.Tensor, positions: torch.Tensor, sigma: float, ref_energy: float = 511.0, n_sigma: float = 1) -> bool:
    summed_energies = energies.sum(dim=1)
    in_window = torch.abs(summed_energies - ref_energy) <= n_sigma * sigma

    mask = in_window

    new_energies = energies[mask]
    new_positions = positions[mask]
    
    return new_energies, new_positions