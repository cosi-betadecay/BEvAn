import itertools

import numpy as np


def classify_tolerance(energies, ref_energy, tolerance):
    """Return True if any energy subset sums within tolerance of ref_energy."""
    for r in range(1, len(energies) + 1):
        for combo in itertools.combinations(energies, r):
            if abs(sum(combo) - ref_energy) < tolerance:
                return True
    return False


def classify_gaussian(energies, ref_energy, sigma, threshold=0.01):
    """Return True if Gaussian likelihood exceeds threshold for any energy subset."""
    for r in range(1, len(energies) + 1):
        for combo in itertools.combinations(energies, r):
            E_sum = sum(combo)
            likelihood = np.exp(-((E_sum - ref_energy) ** 2) / (2 * sigma**2))
            if likelihood > threshold:
                return True
    return False


def classify_confidence(energies, ref_energy, alpha=0.1, k=2.0):
    """Return True if any energy subset falls within k*sigma of ref_energy."""
    for r in range(1, len(energies) + 1):
        for combo in itertools.combinations(energies, r):
            E_sum = sum(combo)
            sigma = alpha * np.sqrt(E_sum)
            if abs(E_sum - ref_energy) < k * sigma:
                return True
    return False
