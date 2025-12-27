"""
Experiment to tune likelihood calculations for annihilation event detection.
Need to create some plots, etc. for analysis.
Combine the ground truth algorithm with the itertools permutations to check effectiveness of likelihoods
for different interaction orderings during real annihilation events.
"""

from typing import Any

import ROOT as M

from beta_decay.mathematics.calculations import calculate_tolerance
from beta_decay.physics.filters import maximum_interaction_distance_filter, verify_compton_angle
from beta_decay.physics.likelihoods import (
    compton_kinematic_likelihood,
    energy_likelihood,
    maximum_interaction_distance_likelihood,
)


def process(event: Any, ref_energy: float) -> int:
    tolerance = calculate_tolerance()
    number_good_events = 0

    for i in range(event.GetNIAs()):
        if event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
            process_id = i + 1

            secondary_ids = []
            for j in range(event.GetNIAs()):
                if event.GetIAAt(j).GetOriginID() == process_id:
                    secondary_ids.append(j + 1)

            total_energy = 0
            for h in range(event.GetNHTs()):
                for sid in secondary_ids:
                    if event.GetHTAt(h).IsOrigin(sid):
                        total_energy += event.GetHTAt(h).GetEnergy()
                        break

            if abs(total_energy - ref_energy) < tolerance:
                _energy_likelihood = energy_likelihood()
                _compton_kin_likelihood = compton_kinematic_likelihood()
                _mid_likelihood = maximum_interaction_distance_likelihood()

                _compton_kin_filter = verify_compton_angle()
                _mid_filter = maximum_interaction_distance_filter()

    return number_good_events
