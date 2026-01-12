import torch


def energy_window(energies: torch.Tensor, sigma: float, ref_energy: float = 511.0, n_sigma=10) -> bool:
    return torch.abs(torch.sum(energies) - ref_energy) <= n_sigma * sigma
