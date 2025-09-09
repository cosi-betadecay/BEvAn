import numpy as np
import logging

def validate_energies_compton(energies: list, positions: list,
                            ref_energy: float = 511.0, tolerance: float =6.0) -> bool:
    m_e_c2 = 511.0

    E_current = ref_energy

    for i in range(len(energies)-1):
        E_meas = energies[i+1]
        r1, r2 = np.array(positions[i]), np.array(positions[i+1])

        v1 = r1 / np.linalg.norm(r1)
        v2 = (r2 - r1) / np.linalg.norm(r2 - r1)

        cos_theta = np.dot(v1, v2)

        E_pred = E_current / (1 + (E_current/m_e_c2)*(1 - cos_theta))

        if abs(E_meas - E_pred) > tolerance:
            return False

        E_current = E_meas

    return True

def validate_energies_compton_v2(energies: list, positions: list,
                            ref_energy: float = 511.0, tolerance: float =6.0,
                            source=np.array([0.0, 0.0, 0.0])) -> bool:
    m_e_c2 = 511.0

    e0, e1 = energies[0], energies[1]
    p0, p1 = np.array(positions[0]), np.array(positions[1])

    v_in = (p0 - source) / np.linalg.norm(p0 - source)
    v_out = (p1 - p0) / np.linalg.norm(p1 - p0)

    cos_theta = np.dot(v_in, v_out)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    e_pred = e0 / (1 + (e0 / m_e_c2) * (1 - cos_theta))

    return abs(e1 - e_pred) <= tolerance

def validate_energies_compton_v3(energies: list, positions: list,
                            ref_energy: float = 511.0, tolerance: float =6.0) -> bool:
    m_e_c2 = 511.0

    e0, e1 = energies[0], energies[1]
    p0, p1 = np.array(positions[0]), np.array(positions[1])

    v_in = (p0) / np.linalg.norm(p0)
    v_out = (p1 - p0) / np.linalg.norm(p1 - p0)

    cos_theta = np.dot(v_in, v_out)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    e_pred = e0 / (1 + (e0 / m_e_c2) * (1 - cos_theta))

    return abs(e1 - e_pred) <= tolerance