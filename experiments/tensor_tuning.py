import itertools
import os
import sys
from typing import Any

import ROOT as M
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mathematics.calculations import calculate_tolerance
from physics.likelihoods.arm import angular_resolution_measure_kernel
from physics.likelihoods.energy import energy_kernel_bdecay
from physics.likelihoods.kn import klein_nishina_pdf
from utils.reader_extraction import get_reader


# --- Prediction loop algorithms ---
def detected_511_event_tensor(
    event: Any,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()
    ref_energy = 511.0

    if n_hits == 0:
        return False

    # Load event data onto GPU
    energies = torch.tensor(
        [event.GetHTAt(i).GetEnergy() for i in range(n_hits)],
        dtype=torch.float32,
        device=device,
    )

    positions = torch.tensor(
        [
            (
                event.GetHTAt(i).GetPosition().X(),
                event.GetHTAt(i).GetPosition().Y(),
                event.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Build all hit combinations (CPU once, GPU afterwards)
    combos = []
    sizes = []

    for r in range(1, n_hits + 1):
        for c in itertools.combinations(range(n_hits), r):
            combos.append(c)
            sizes.append(r)

    max_r = max(sizes)
    n_combo = len(combos)

    # Pad combinations with -1
    idx = torch.full((n_combo, max_r), -1, dtype=torch.long)
    for i, c in enumerate(combos):
        idx[i, : len(c)] = torch.tensor(c)

    idx = idx.to(device)
    sizes = torch.tensor(sizes, device=device)

    # idx: (n_combo, max_r), padded with -1

    energy_combo = energies[idx]  # (n_combo, max_r)
    pos_combo = positions[idx]  # (n_combo, max_r, 3)

    mask = idx >= 0
    energy_combo = torch.where(mask, energy_combo, torch.zeros_like(energy_combo))
    pos_combo = torch.where(mask[..., None], pos_combo, torch.zeros_like(pos_combo))

    return posterior_torch(energy_combo, pos_combo, ref_energy, tolerance, 1.0, 1.0, 0.0, sizes)


def detected_511_event(
    event: Any,
) -> bool:
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()
    ref_energy = 511.0

    energies = torch.tensor([event.GetHTAt(i).GetEnergy() for i in range(n_hits)], dtype=torch.float32)
    positions = torch.tensor(
        [
            (
                event.GetHTAt(i).GetPosition().X(),
                event.GetHTAt(i).GetPosition().Y(),
                event.GetHTAt(i).GetPosition().Z(),
            )
            for i in range(n_hits)
        ],
        dtype=torch.float32,
    )

    best = -float("inf")

    for r in range(1, n_hits + 1):
        for idx_combo in itertools.combinations(range(n_hits), r):
            energy_combo = energies[list(idx_combo)]
            pos_combo = positions[list(idx_combo)]

            score = posterior(energy_combo, pos_combo, ref_energy, tolerance, 1.0, 1.0, 0.0, r)
            best = max(best, score)

    return best


# --- Posteriors ---
def posterior_torch(
    energy_combo: torch.Tensor,
    pos_combo: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
    sizes: torch.Tensor,
) -> float:
    device = energy_combo.device
    n_combo = energy_combo.shape[0]
    one = torch.ones(n_combo, device=device)

    # ARM
    arm = one.clone()
    valid_arm = sizes > 2
    if valid_arm.any():
        arm[valid_arm] = angular_resolution_measure_kernel(
            energy_combo[valid_arm],
            pos_combo[valid_arm],
        )

    # KN
    kn = one.clone()
    valid_kn = sizes > 1
    if valid_kn.any():
        kn[valid_kn] = klein_nishina_pdf(energy_combo[valid_kn])

    # Energy
    eng = energy_kernel_bdecay(energy_combo, ref_energy, tolerance)
    # Numerical stability
    # arm = torch.clamp(arm, min=1e-12)
    # kn = torch.clamp(kn, min=1e-12)
    # eng = torch.clamp(eng, min=1e-12)

    # score = alpha_energy * eng
    # score = alpha_arm * arm # valid
    score = alpha_energy * eng + alpha_arm * torch.log(arm)

    return score.max()


def posterior(
    energies: torch.Tensor,
    positions: torch.Tensor,
    ref_energy: float,
    tolerance: float,
    alpha_energy: float,
    alpha_arm: float,
    alpha_kn: float,
    r: int,
) -> float:
    one = torch.tensor(1.0, dtype=energies.dtype, device=energies.device)

    eng = energy_kernel_bdecay(energies, ref_energy, tolerance)
    arm = angular_resolution_measure_kernel(energies, positions) if r > 2 else one
    kn = klein_nishina_pdf(energies) if r > 1 else one

    # arm = torch.clamp(arm, min=1e-12)
    # kn = torch.clamp(kn, min=1e-12)
    # eng = torch.clamp(eng, min=1e-12)

    # score = alpha_energy * eng
    # score = alpha_arm * arm # valid

    score = alpha_energy * eng + alpha_arm * torch.log(arm)

    return score.max()


# --- Simulation ---
def annihilation_extractor(
    geometry_file: str,
    sim_file: str,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    i = 1
    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit=" events",
    ):
        M.SetOwnership(event, True)
        if event.GetNHTs() > 2:
            s_t = detected_511_event_tensor(event)
            s = detected_511_event(event)
            print(f"Iteration {i}")
            print("Scores: ", s_t, s)
            if s_t != s:
                print("pwfwågwegwegwåegwegewågweå")
            print("-" * 50)
            i += 1
        if i == 100:
            break


if __name__ == "__main__":
    annihilation_extractor(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )
