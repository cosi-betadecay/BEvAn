import random as rd

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class SyntheticDataGenerator:
    """
    Synthetic Monte Carlo generator for gamma-ray interaction sequences based on
    simplified Compton scattering physics.

    This class generates variable-length hit sequences representing photon
    interactions in a detector volume. Each event consists of a sequence of
    Compton scatters followed by a final absorption, with energies and directions
    sampled according to the Klein–Nishina differential cross section.

    The generator is intended for:
    - Likelihood diagnostics
    - Posterior visualization
    - Stress-testing reconstruction and inference pipelines
    - Controlled studies of energy and angular response

    Physics assumptions:
    - Single-photon events (no pile-up)
    - Free-electron Compton scattering
    - No Doppler broadening or binding effects
    - Straight-line propagation between interactions
    - Final interaction is full absorption

    The Compton scattering angle is sampled using rejection sampling from the
    Klein–Nishina formula, while the azimuthal angle is assumed uniform.

    References
    ----------
    - Klein, O., Nishina, Y. Über die Streuung von Strahlung durch freie Elektronen nach der neuen relativistischen Quantendynamik von Dirac. Z. Physik 52, 853–868 (1929). https://doi.org/10.1007/BF01366453
    - Lei, F., Dean, A.J. & Hills, G.L. Compton Polarimetry in Gamma-Ray Astronomy. Space Science Reviews 82, 309–388 (1997). https://doi.org/10.1023/A:1005027107614
    """

    def __init__(
        self,
        max_hits_per_sequence: int,
        seed: int = 0,
    ):
        self.max_hits = max_hits_per_sequence
        self.MEC2 = 511.0  # keV

        rd.seed(seed)
        torch.manual_seed(seed)

    def sample_theta_kn(self, E: float) -> float:
        """
        Sample a Compton scattering polar angle from the Klein–Nishina distribution.

        The scattering angle θ is sampled using rejection sampling from the
        Klein–Nishina differential cross section for unpolarized photons:

            dσ/dΩ ∝ (E'/E)^2 [E'/E + E/E' − sin²θ]

        where E is the incident photon energy and E' is the scattered photon energy.

        The azimuthal angle is not sampled here and is assumed uniform elsewhere.

        Args:
            E (float): Incident photon energy in keV.

        Returns:
            float: Scattering polar angle θ in radians.
        """
        mec2 = self.MEC2
        w_max = 2.0 + 2.0 * (E / self.MEC2)

        while True:
            cos_theta = rd.uniform(-1.0, 1.0)
            theta = np.arccos(cos_theta)

            E_prime = E / (1.0 + (E / mec2) * (1.0 - cos_theta))
            ratio = max(E_prime / E, 1e-12)

            weight = ratio**2 * (ratio + 1.0 / ratio - (1.0 - cos_theta**2))

            if rd.uniform(0.0, w_max) < weight:
                return theta

    def sample_initial_energy(self, mode: int) -> float:
        """
        Sample an initial photon energy according to the selected generation mode.

        Modes:
            0 : Mixed population of events, including annihilation-like,
                low-energy background, medium-energy background, and rare
                high-energy events.
            1 : True annihilation events (monoenergetic 511 keV photons).
            2 : False / background events with energies away from 511 keV.

        The distributions are heuristic but chosen to reflect realistic
        gamma-ray detector environments.

        Args:
            mode (int): Event generation mode (0 = all, 1 = true, 2 = false).

        Raises:
            ValueError: If an unknown mode is specified.

        Returns:
            float: Initial photon energy in keV.
        """
        if mode == 0:
            r = rd.random()
            if r < 0.4:
                return max(1.0, rd.gauss(self.MEC2, 20.0))
            elif r < 0.7:
                return rd.uniform(50.0, 500.0)
            elif r < 0.9:
                return rd.uniform(500.0, 3000.0)
            else:
                return rd.uniform(3000.0, 30000.0)

        elif mode == 1:
            return self.MEC2

        elif mode == 2:
            r = rd.random()
            return rd.uniform(50.0, 450.0) if r < 0.7 else rd.uniform(600.0, 5000.0)

        else:
            raise ValueError(f"Unknown mode {mode}")

    def compton_step(self, E: float, prev_pos: torch.Tensor) -> tuple[float, float, torch.Tensor]:
        """
        Perform a single Compton scattering step.

        Given an incident photon energy and position, this function:
        - Samples a Compton scattering angle from the Klein–Nishina distribution
        - Samples a uniform azimuthal angle
        - Computes the scattered photon energy using the Compton formula
        - Advances the photon position along the new direction

        Args:
            E (float): Incident photon energy in keV.
            prev_pos (torch.Tensor): Previous photon position in 3D space.

        Returns:
            tuple[float, float, torch.Tensor]: Scattered energy loss (dE), new energy (E_new), and new position (new_pos).
        """
        theta = self.sample_theta_kn(E)
        phi = rd.uniform(0.0, 2 * np.pi)

        E_new = E / (1.0 + (E / self.MEC2) * (1.0 - np.cos(theta)))
        dE = E - E_new

        direction = torch.tensor(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

        new_pos = prev_pos + direction
        return dE, E_new, new_pos

    def generate_single_event(self, E0: float, n_hits: int, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single photon interaction sequence.

        The event consists of (n_hits - 1) Compton scatterings followed by a final
        absorption that deposits the remaining energy. Positions are generated
        sequentially assuming straight-line propagation between interactions.

        For background events (mode = 2), the order of energy deposits is randomized
        to intentionally break physical ordering assumptions.

        Args:
            E0 (float): Initial photon energy in keV.
            n_hits (int): Number of interaction hits in the event.
            mode (int): Event generation mode.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Energies and positions of the interaction hits.
        """
        event_energies = []
        event_positions = []

        E = E0
        pos = torch.zeros(3)

        for _ in range(n_hits - 1):
            dE, E, new_pos = self.compton_step(E, pos)
            event_energies.append(dE)
            event_positions.append(pos)
            pos = new_pos

        # final absorption
        event_energies.append(E)
        event_positions.append(pos)

        energies = torch.tensor(event_energies)
        positions = torch.stack(event_positions)

        if mode == 2:
            perm = torch.randperm(n_hits)
            energies = energies[perm]

        return energies, positions

    def generate(self, num_samples: int, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of synthetic photon interaction events.

        Each event has a random number of hits between 2 and `max_hits_per_sequence`.
        Events are padded with zeros to form rectangular tensors suitable for
        vectorized processing and neural network input.

        A progress bar is displayed during generation.

        Args:
            num_samples (int): Number of events to generate.
            mode (int): Event generation mode.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Energies and positions of all generated events.
        """
        energies = []
        positions = []

        iterator = range(num_samples)
        iterator = tqdm(iterator, desc="Generating synthetic events")

        for _ in iterator:
            n_hits = rd.randint(2, self.max_hits)
            E0 = self.sample_initial_energy(mode)
            e, p = self.generate_single_event(E0, n_hits, mode)
            energies.append(e)
            positions.append(p)

        energies = pad_sequence(energies, batch_first=True, padding_value=0.0)
        positions = pad_sequence(positions, batch_first=True, padding_value=0.0)

        return energies, positions
