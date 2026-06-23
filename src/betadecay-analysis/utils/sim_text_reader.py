"""Pure-Python reader for MEGAlib .sim files (format Version 101).

Parses the text format directly — no ROOT / MEGAlib build required — and
exposes exactly the per-event fields this project consumes: measured hits
(position, energy, origin IA ids) and the truth interaction (IA) records.
Field layout follows the verified spec in
``.claude/skills/megalib/references/sim-format.md`` (parsing code ground
truth: MSimIA::AddRawInput, MSimHT::AddRawInput in MEGAlib's src/sivan/).

Energies are read verbatim as cosima wrote them — the same values
``MSimHT::GetEnergy`` returns when no additional noising is applied on load,
which is how the pipeline currently reads them.

``event_is_bdecay`` replicates ``physics.ground_truths.ground_truth_bdecay``
(same one-level secondary lookup, same per-hit single counting) so labels
match the production pipeline.

Units: cm, keV, seconds.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class SimIA:
    """One IA truth record (subset of the 24 fields)."""

    process: str  # "INIT", "DECA", "ANNI", "COMP", "PHOT", ...
    ia_id: int  # 1-based interaction ID
    origin_id: int  # IA ID that created the incoming particle (0 = primary)
    position: tuple[float, float, float]
    secondary_pid: int
    secondary_dir: tuple[float, float, float]
    secondary_energy: float  # keV


@dataclass
class SimHit:
    """One HTsim measured hit."""

    detector: int
    position: tuple[float, float, float]
    energy: float  # keV
    origins: tuple[int, ...]  # IA ids that contributed energy to this hit


@dataclass
class SimTextEvent:
    """One parsed SE block: its event id, truth IA records, and measured hits."""

    event_id: int
    ias: list[SimIA] = field(default_factory=list)
    hits: list[SimHit] = field(default_factory=list)


def parse_ia(line: str) -> SimIA:
    """Parse one ``IA`` truth line into a :class:`SimIA`."""
    head, rest = line[3:].split(maxsplit=1)
    fields = rest.split(";")
    # fields: [0]=ID [1]=origin [2]=detector [3]=time [4..6]=x,y,z
    # [7]=mother PID [8..10]=mother dir [11..13]=mother pol [14]=mother E
    # [15]=secondary PID [16..18]=sec dir [19..21]=sec pol [22]=sec E
    return SimIA(
        process=head,
        ia_id=int(fields[0]),
        origin_id=int(fields[1]),
        position=(float(fields[4]), float(fields[5]), float(fields[6])),
        secondary_pid=int(fields[15]),
        secondary_dir=(float(fields[16]), float(fields[17]), float(fields[18])),
        secondary_energy=float(fields[22]),
    )


def parse_ht(line: str) -> SimHit:
    """Parse one ``HTsim`` measured-hit line into a :class:`SimHit`."""
    fields = line.split(maxsplit=1)[1].split(";")
    # fields: [0]=detector [1..3]=x,y,z [4]=energy [5]=time [6..]=origin IAs
    return SimHit(
        detector=int(fields[0]),
        position=(float(fields[1]), float(fields[2]), float(fields[3])),
        energy=float(fields[4]),
        origins=tuple(int(f) for f in fields[6:] if f.strip()),
    )


def iter_sim_events(path: str) -> Iterator[SimTextEvent]:
    """Yield SimTextEvent for every SE block in the file, in order."""
    event = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("SE"):
                if event is not None:
                    yield event
                event = SimTextEvent(event_id=-1)
            elif event is None:
                continue  # still in the file header
            elif line.startswith("ID"):
                event.event_id = int(line.split()[1])
            elif line.startswith("IA"):
                event.ias.append(parse_ia(line))
            elif line.startswith("HTsim"):
                event.hits.append(parse_ht(line))
    if event is not None:
        yield event


def event_is_bdecay(event: SimTextEvent, tolerance: float) -> bool:
    """Replicates physics.ground_truths.ground_truth_bdecay on text records:
    the event is signal iff some ANNI photon's direct secondaries deposit a
    summed hit energy within ``tolerance`` of 511 keV. Each hit counts
    once (first matching secondary), matching the production loop's break.
    """
    for ia in event.ias:
        if ia.process != "ANNI":
            continue
        secondary_ids = {other.ia_id for other in event.ias if other.origin_id == ia.ia_id}
        total = 0.0
        for hit in event.hits:
            if any(o in secondary_ids for o in hit.origins):
                total += hit.energy
        if abs(total - 511.0) < tolerance:
            return True
    return False
