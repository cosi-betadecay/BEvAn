import os
import sys

import omegaconf
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mathematics.geometry import theta_geo, theta_kin


def angular_resolution_measure_kernel(
    energies: torch.Tensor, positions: torch.Tensor, cfg_likelihoods: omegaconf.dictconfig.DictConfig
) -> torch.Tensor:
    _theta_geo, _sigma_theta_geo = theta_geo(positions, cfg_likelihoods)
    _theta_kin, _sigma_theta_kin = theta_kin(energies, cfg_likelihoods)

    arm = _theta_geo - _theta_kin
    var_arm = _sigma_theta_geo**2 + _sigma_theta_kin**2
    var_arm = torch.clamp(var_arm, min=1e-6)

    return torch.exp(-(arm**2 / var_arm) / 2)


# Add for bg too
