def detected_511_event(ref_energy, tolerance, Event):
    energy_sums = {}

    n_hits = Event.GetNHTs()
    for i in range(n_hits):
        energy = Event.GetHTAt(i).GetEnergy()

        for j in range(Event.GetHTAt(i).GetNOrigins()):
            origin_id = Event.GetHTAt(i).GetOriginAt(j)
            if origin_id not in energy_sums:
                energy_sums[origin_id] = 0.0
            energy_sums[origin_id] += energy

    logging.info(energy_sums)

    for total in energy_sums.values():
        if abs(total - ref_energy) < tolerance:
            return True

    return False


def detected_511_event_v2(ref_energy, tolerance, Event):
    n_hits = Event.GetNHTs()

    # One
    for i in range(n_hits):
        e = Event.GetHTAt(i).GetEnergy()
        if abs(e - ref_energy) < tolerance:
            return True

    # Two
    for i in range(n_hits):
        e1 = Event.GetHTAt(i).GetEnergy()
        for j in range(i+1, n_hits):
            e2 = Event.GetHTAt(j).GetEnergy()
            if abs((e1 + e2) - ref_energy) < tolerance:
                return True

    return False