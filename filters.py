import numpy as np
import logging
from collections import defaultdict

def validate_energies_compton(energies: list, positions: list,
                            ref_energy: float = 511.0, tolerance: float =6.0) -> bool:

    r1, r2, r3 = np.array(positions[0]), np.array(positions[1]), np.array(positions[2])

    v1 = r2 - r1
    v2 = r3 - r1

    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)

    tetta = np.arccos(dot)/(norms)

    if np.isnan(tetta):
        return False

    return True


# Pixel sizes (from .geo.setup)
PIXEL_SIZE_X = 0.2     # cm
PIXEL_SIZE_Y = 0.2     # cm
PIXEL_SIZE_Z = 0.0375  # cm

def get_pixel_indices(pos):
    x, y, z = pos
    return (int(x / PIXEL_SIZE_X),
            int(y / PIXEL_SIZE_Y),
            int(z / PIXEL_SIZE_Z))

def cluster_hits(positions, energies):
    clusters = defaultdict(list)
    
    for pos, E in zip(positions, energies):
        pix = get_pixel_indices(pos)
        clusters[pix].append((pos, E))
    
    clustered_positions = []
    clustered_energies = []
    
    for pix, hits in clusters.items():
        total_E = sum(h[1] for h in hits)
        clustered_energies.append(total_E)
        
        weights = np.array([h[1] for h in hits])
        coords = np.array([h[0] for h in hits])
        avg_pos = np.average(coords, axis=0, weights=weights)
        clustered_positions.append(tuple(avg_pos))
    
    return clustered_positions, clustered_energies
    
    


"""
1) compton cone check?
2) Klein-Nishina?
"""