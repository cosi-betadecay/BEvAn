import omegaconf
import torch


def maximum_interaction_distance_filter(
    positions: torch.Tensor, cfg_preprocessing: omegaconf.dictconfig.DictConfig
) -> bool:
    if positions.shape[0] < 2:
        return True

    diffs = positions[1:] - positions[:-1]  # shape (N-1, 3)
    distances = torch.norm(diffs, dim=1)  # shape (N-1,)

    if torch.isnan(distances).any():
        return False

    return bool(torch.all(distances <= cfg_preprocessing.mid.mid_threshold))


def mid_preprocessing(
    energies: torch.Tensor, positions: torch.Tensor, cfg_preprocessing: omegaconf.dictconfig.DictConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = torch.tensor(
        [
            maximum_interaction_distance_filter(positions[i], cfg_preprocessing)
            for i in range(positions.shape[0])
        ],
        device=energies.device,
        dtype=torch.bool,
    )
    valid_energies = energies[valid_mask]
    valid_positions = positions[valid_mask]

    return valid_energies, valid_positions
