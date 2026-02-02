import random as rd

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class SyntheticDataGenerator:
    def __init__(
        self,
        max_hits_per_sequence: int,
        seed: int = 0,
    ):
        self.max_hits = max_hits_per_sequence
        self.MEC2 = 511.0  # keV

        rd.seed(seed)
        torch.manual_seed(seed)

    def generate(
        self,
        num_samples: int,
        mode: bool,
    ):
        energies = []
        positions = []

        for _ in range(num_samples):
            n_hits = rd.randint(2, self.max_hits)
            if mode:
                E_remaining = self.MEC2
            else:
                low = rd.uniform(0.0, 490.0)
                high = rd.uniform(530.0, 10000.0)
                E_remaining = low if rd.random() < 0.5 else high

            event_energies = []
            event_positions = []

            prev_pos = torch.zeros(3)

            for _ in range(n_hits - 1):
                theta = rd.uniform(0.0, np.pi / 2)
                phi = rd.uniform(0.0, 2 * np.pi)

                E_new = E_remaining / (1 + (E_remaining / self.MEC2) * (1 - np.cos(theta)))
                dE = E_remaining - E_new

                new_dir = torch.tensor(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )

                new_pos = prev_pos + new_dir

                event_energies.append(dE)
                event_positions.append(prev_pos)

                prev_pos = new_pos
                E_remaining = E_new

            # final absorption
            event_energies.append(E_remaining)
            event_positions.append(prev_pos)

            event_energies = torch.tensor(event_energies)
            event_positions = torch.stack(event_positions)

            if not mode:
                perm = torch.randperm(n_hits)
                event_energies = event_energies[perm]

            energies.append(event_energies)
            positions.append(event_positions)

        energies = pad_sequence(energies, batch_first=True, padding_value=0.0)
        positions = pad_sequence(positions, batch_first=True, padding_value=0.0)

        return energies, positions
