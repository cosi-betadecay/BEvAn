import numpy as np
import torch


def klein_nishina_likelihood(energies: torch.Tensor) -> float:
    """Calculate the Klein–Nishina weight for a given set of energy deposits.

    Args:
        energies (torch.Tensor): Tensor of energy deposits.

    Returns:
        float: Klein–Nishina weight.
    """

    def cos_theta(E_0: float, E: float, electron_mass_energy: float) -> float:
        return 1 - electron_mass_energy * (1 / E - 1 / E_0)

    def kn_log_likelihood(energies: torch.Tensor) -> float:
        r_e = 2.8179403227e-15  # classical electron radius in meters
        electron_mass_energy = 511.0  # keV
        E_0 = energies.sum()
        E = E_0 - energies[0]

        _cos_theta = cos_theta(E_0, E, electron_mass_energy)

        lam_ratio = E / E_0
        theta = torch.acos(_cos_theta)

        r_e = 2.8179403227e-15  # classical electron radius, m

        # Klein-Nishina differential cross-section (unpolarized)
        kn = 0.5 * (r_e**2) * (lam_ratio**2) * (lam_ratio + (1 / lam_ratio) - torch.sin(theta) ** 2)
        log_kn = torch.log(kn + 1e-40)

        kn_max = 0.5 * (r_e**2)
        log_kn_max = np.log(kn_max + 1e-40)

        z = log_kn - log_kn_max

        return torch.exp(-z**2)
    
    if len(energies) < 2:
        return 0.0

    return float(kn_log_likelihood(energies))
