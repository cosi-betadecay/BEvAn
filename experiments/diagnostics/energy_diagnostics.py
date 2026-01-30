import itertools
import os
import sys
from typing import Any

import ROOT as M
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mathematics.calculations import calculate_tolerance
from physics.annihilation_detection import ground_truth
from physics.likelihoods.energy import energy_kernel_bdecay
from utils.plots import energy_kernel_vs_total_energy
from utils.reader_extraction import get_reader


def detected_511_event(
    ref_energy: float,
    event: Any,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tolerance = calculate_tolerance()
    n_hits = event.GetNHTs()
    ref_energy = 511.0

    if n_hits == 0:
        return None, None

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

    eng = energy_kernel_bdecay(energy_combo, ref_energy, tolerance)
    total_energy = torch.sum(energy_combo, dim=1)
    valid = torch.isfinite(eng) & torch.isfinite(total_energy) & (total_energy > 0)

    eng = eng[valid].detach().cpu()
    total_energy = total_energy[valid].detach().cpu()

    return eng, total_energy


def annihilation_extractor_test_kn(
    geometry_file: str,
    sim_file: str,
    ref_energy: int = 511,
) -> None:
    reader = get_reader(geometry_file, sim_file)

    combs = []
    sums = []
    ground_truths = []

    for event in tqdm(
        iter(lambda: reader.GetNextEvent(), None),
        desc="Processing events",
        unit="event",
    ):
        M.SetOwnership(event, True)

        is_annihilation = ground_truth(event, ref_energy)
        if is_annihilation:
            ground_truths.append(1)
        else:
            ground_truths.append(0)

        _comb, _sum = detected_511_event(
            ref_energy,
            event,
        )

        if _comb is None:
            _comb = torch.empty(0)
            _sum = torch.empty(0)

        combs.append(_comb)
        sums.append(_sum)

    return combs, sums, ground_truths


##################################################################################


if __name__ == "__main__":
    load_dotenv()

    (
        combs,
        sums,
        ground_truths,
    ) = annihilation_extractor_test_kn(
        "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup", "data/Activation.sim"
    )

    true_events_energy = []
    false_events_energy = []
    sums_true_events = []
    sums_false_events = []

    for i in range(len(ground_truths)):
        if ground_truths[i] == 1:
            true_events_energy.append(combs[i])
            sums_true_events.append(sums[i])
        else:
            false_events_energy.append(combs[i])
            sums_false_events.append(sums[i])

    runs = [
        ("energy", combs, sums),
        ("energy_true_events", true_events_energy, sums_true_events),
        ("energy_false_events", false_events_energy, sums_false_events),
    ]

    for label, data, sums in runs:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
        wandb.init(project="cosi-betadecay-energy-diagnostics", name=label, reinit=True)

        data_tensor = torch.cat(data) if len(data) > 0 else torch.empty(0)
        sums_tensor = torch.cat(sums) if len(sums) > 0 else torch.empty(0)
        energy_kernel_vs_total_energy(data_tensor, sums_tensor, label)

        wandb.finish()
