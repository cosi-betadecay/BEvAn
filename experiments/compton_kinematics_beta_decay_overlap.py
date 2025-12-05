import itertools
from typing import Any

import ROOT as M
import torch
import tqdm
from src.beta_decay.utils.reader_extraction import get_reader

# Beta decay detection
def detected_511_event(
    ref_energy: float, Event: Any, tolerance: float, n_hits: int, energies: torch.Tensor
) -> bool:
    """Check if event contains a hit combination summing to the reference energy.

    Args:
        ref_energy (float): Reference energy to compare against (e.g., 511 keV).
        Event (Any): MEGAlib event object with detector hits.
        tolerance (float): Allowed energy deviation from the reference.

    Returns:
        bool: True if a hit combination matches the reference energy, else False.
    """
    for r in range(1, n_hits + 1):
        for combo in itertools.combinations(energies, r):
            combo_tensor = torch.stack(combo)

            if torch.abs(combo_tensor.sum() - ref_energy) < tolerance:
                return True

    return False


# Compton kinematics filter
def verify_compton_angle(energies: torch.Tensor) -> bool:
    """Calculate the Compton scattering angle from two energy deposits.

    Args:
        energies (list): List of energy values for a Compton sequence.

    Returns:
        bool: True if the angle is physically valid, False otherwise.
    """
    if energies.numel() < 2:
        return False  # For pure kinematic validation

    def sigma_cos_phi(
        E_0: torch.Tensor, E: torch.Tensor, frac_sigma: float = 0.0035
    ) -> float | None:
        """Calculate the uncertainty in the cosine of the Compton scattering angle.

        Args:
            E_0 (torch.Tensor): Sum of all energy deposits.
            E (torch.Tensor): Sum of all but the first energy deposit.
            frac_sigma (float): Fractional uncertainty. Defaults to 0.35% (due to HPGe detector resolution).

        Returns:
            float: Uncertainty in the cosine of the Compton scattering angle.
        """
        electron_mass_energy = 511.0  # keV

        sigma_E_0 = frac_sigma * E_0
        sigma_E = frac_sigma * E

        term0 = (sigma_E_0 / (E_0**2)) ** 2
        term1 = (sigma_E / (E**2)) ** 2

        return electron_mass_energy * torch.sqrt(term0 + term1)

    E_0 = energies.sum()
    E = energies[1:].sum()

    electron_mass_energy = 511.0  # keV
    cos_phi = 1 - electron_mass_energy * (1 / E - 1 / E_0)

    _sigma_cos_phi = sigma_cos_phi(E_0, E)

    limit = 1.0 + _sigma_cos_phi

    return bool(torch.abs(cos_phi) <= limit)


def main():
    reader = get_reader(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "Activation.sim"
    )

    for Event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(Event, True)

        n_hits = Event.GetNHTs()

        energies = torch.tensor(
            [Event.GetHTAt(i).GetEnergy() for i in range(n_hits)], dtype=torch.float32
        )

        compton_valid = any(
            verify_compton_angle(torch.tensor(list(combo), dtype=torch.float32))
            for combo in itertools.combinations(energies, 2)
        )
        beta_decay_valid = detected_511_event(
            ref_energy=511.0,
            Event=Event,
            tolerance=5.0,
            n_hits=n_hits,
            energies=energies,
        )


if __name__ == "__main__":
    main()
