import itertools

import numpy as np


def compton_cone_filter(energies: list, positions: list) -> bool:
    """Return True if any hit pair passes the Compton cone annihilation filter.

    Args:
        energies (list): List of energy values for each hit.
        positions (list): List of position vectors for each hit.

    Returns:
        bool: True if any hit pair passes the Compton cone filter, False otherwise.
    """
    if len(energies) < 2:
        return False

    for hit_pair in itertools.combinations(range(len(energies)), 2):
        E1 = energies[hit_pair[0]]
        E2 = energies[hit_pair[1]]

        if E1 + E2 <= 511.0:
            continue

        cos_theta = 1 - 511.0 / E2 + 511.0 / (E1 + E2)
        if cos_theta < -1.0 or cos_theta > 1.0:
            continue

        theta = np.arccos(cos_theta)

        pos1 = positions[hit_pair[0]]
        pos2 = positions[hit_pair[1]]
        vec = pos2 - pos1
        distance = np.linalg.norm(vec)
        if distance == 0:
            continue
        unit_vec = vec / distance

        cone_angle = theta

        if abs(cone_angle - np.pi / 2) < 0.1:
            return True

    return False
