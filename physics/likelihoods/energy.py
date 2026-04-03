import torch


def energy_kernel_bdecay(energies: torch.Tensor, ref_energy: float, sigma_e: float) -> float:
    E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    kernel = torch.exp(-(((E - ref_energy) / (10 * sigma_e)) ** 2) / 2)

    return torch.clamp(kernel, min=1e-12)
