import omegaconf
import torch


def energy_window(
    energies: torch.Tensor,
    positions: torch.Tensor,
    sigma: float,
    cfg_preprocessing: omegaconf.dictconfig.DictConfig,
) -> bool:
    n_std = cfg_preprocessing.energy.n_std

    summed_energies = energies.sum(dim=1)
    in_window = torch.abs(summed_energies - 511) <= n_std * sigma

    mask = in_window

    new_energies = energies[mask]
    new_positions = positions[mask]

    return new_energies, new_positions
