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

def detected_511_event(ref_energy, tolerance, Event):
    n_hits = Event.GetNHTs()
    energies = [Event.GetHTAt(i).GetEnergy() for i in range(n_hits)]
    positions = [(Event.GetHTAt(i).GetPosition().X(),
                  Event.GetHTAt(i).GetPosition().Y(),
                  Event.GetHTAt(i).GetPosition().Z())
                 for i in range(n_hits)]

    # Konstant: elektron hvilemasse i keV
    m_e_c2 = 511.0  

    # Vi sjekker alle kombinasjoner av hits som kan summere til 511 keV
    for r in range(2, n_hits+1):  # minst to hits for å få en vinkel
        for combo_indices in itertools.combinations(range(n_hits), r):
            combo_energies = [energies[i] for i in combo_indices]
            combo_positions = [positions[i] for i in combo_indices]

            if abs(sum(combo_energies) - ref_energy) < tolerance:
                prev_vec = None
                # --- Compton-fysikk-sjekk ---
                # Vi går parvis gjennom scatteringene
                for i in range(len(combo_positions)-1):
                    p1 = np.array(combo_positions[i])
                    p2 = np.array(combo_positions[i+1])

                    r_vec = p2 - p1
                    if np.linalg.norm(r_vec) == 0:
                        continue
                    r_vec = r_vec / np.linalg.norm(r_vec)  # enhetsvektor

                    if i > 0:  # sammenlign med forrige retning
                        cos_theta = np.dot(prev_vec, r_vec)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # stabilitet
                        theta = np.arccos(cos_theta)

                        # Energien før og etter scatter
                        E_in = combo_energies[i]
                        E_out = combo_energies[i+1]

                        # Teoretisk forventet E' fra Compton-formelen
                        E_expected = E_in / (1 + (E_in/m_e_c2)*(1 - np.cos(theta)))

                        if abs(E_out - E_expected) > 20:  # 20 keV vindu, kan tunes
                            return False  # ikke konsistent

                    prev_vec = r_vec

                # Hvis vi kom hit, så er både summen ~511 og Compton-sjekken OK
                return True

    return False


def validate_energies_compton(energies, positions, ref_energy=511.0, tolerance=6.0):
    m_e_c2 = 511.0

    E_current = ref_energy

    for i in range(len(energies)-1):
        E_meas = energies[i+1]
        r1, r2 = np.array(positions[i]), np.array(positions[i+1])

        logging.info("r1 = %s", r1)
        logging.info("r2 = %s", r2)

        v1 = r1 / np.linalg.norm(r1)
        v2 = (r2 - r1) / np.linalg.norm(r2 - r1)

        logging.info("v1 = %s", v1)
        logging.info("v2 = %s", v2)

        cos_theta = np.dot(v1, v2)

        logging.info("cos_theta = %s", cos_theta)

        E_pred = E_current / (1 + (E_current/m_e_c2)*(1 - cos_theta))
        logging.info("Energy pred = %s", E_pred)
        logging.info("Energy actual = %s", E_meas)

        logging.info("-"*100)

        if abs(E_meas - E_pred) > tolerance:
            return False

        E_current = E_meas

    return True