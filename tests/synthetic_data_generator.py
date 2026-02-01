import random as rd

import torch


def synthetic_data_generator(
    num_samples: int,
    max_hits_per_sequence: int,
    mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    aaa = rd.randint(0, 100)
    return
