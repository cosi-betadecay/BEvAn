from __future__ import annotations

import math
import os
from pathlib import Path
from typing import ClassVar


class MString(str):
    """MEGAlib string: plain ``str`` semantics are all the pipeline uses."""

    __slots__ = ()


def SetOwnership(obj, flag) -> None:
    """PyROOT memory-management helper; nothing to manage here."""
    return None


class _GSystem:
    def Load(self, path) -> int:
        return 0


class _GInterpreter:
    def Declare(self, code) -> bool:
        return True


gSystem = _GSystem()
gInterpreter = _GInterpreter()


class MGlobal:
    def Initialize(self) -> bool:
        return True


class MDGeometryQuest:
    """Fake geometry: records the path; 'scans' successfully iff the file exists."""

    def __init__(self) -> None:
        self.path: str | None = None

    def ScanSetupFile(self, path) -> bool:
        self.path = str(path)
        return Path(os.path.expandvars(self.path)).is_file()


############################################
# .sim side
############################################


class MVector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._x, self._y, self._z = float(x), float(y), float(z)

    def X(self) -> float:
        return self._x

    def Y(self) -> float:
        return self._y

    def Z(self) -> float:
        return self._z


class SimHit:
    """One ``HTsim`` line: detector, position, energy, and origin IA ids."""

    def __init__(self, position: tuple[float, float, float], energy: float, origins: tuple[int, ...] = ()):
        self._position = MVector(*position)
        self._energy = float(energy)
        self._origins = tuple(int(o) for o in origins)

    def GetPosition(self) -> MVector:
        return self._position

    def GetEnergy(self) -> float:
        return self._energy

    def IsOrigin(self, ia_id: int) -> bool:
        return int(ia_id) in self._origins


class SimIA:
    """One ``IA`` line: process name, positional id, and origin IA id."""

    def __init__(self, process: str, ia_id: int, origin_id: int):
        self._process = MString(process)
        self._id = int(ia_id)
        self._origin_id = int(origin_id)

    def GetProcess(self) -> MString:
        return self._process

    def GetId(self) -> int:
        return self._id

    def GetOriginID(self) -> int:
        return self._origin_id


class SimEvent:
    """One ``SE`` block of a ``.sim`` file (also a synthetic-event builder)."""

    def __init__(self, event_id: int, hits: list[SimHit] | None = None, ias: list[SimIA] | None = None):
        self._id = int(event_id)
        self._hits = list(hits or [])
        self._ias = list(ias or [])

    def GetID(self) -> int:
        return self._id

    def GetNHTs(self) -> int:
        return len(self._hits)

    def GetHTAt(self, i: int) -> SimHit:
        return self._hits[i]

    def GetNIAs(self) -> int:
        return len(self._ias)

    def GetIAAt(self, i: int) -> SimIA:
        return self._ias[i]


def _parse_ia_line(line: str) -> SimIA:
    # "IA COMP 12;10;3;<time>;<x>;<y>;<z>;..." -> process COMP, id 12, origin 10.
    head, *rest = line[3:].split(";")
    process, ia_id = head.split()
    return SimIA(process, int(ia_id), int(rest[0]))


def _parse_ht_line(line: str) -> SimHit:
    # "HTsim 3; x; y; z; energy; time; origin;origin;..." (origins optional).
    parts = line[6:].split(";")
    x, y, z, energy = (float(p) for p in parts[1:5])
    origins = tuple(int(p) for p in parts[6:] if p.strip())
    return SimHit((x, y, z), energy, origins)


def iter_sim_events(path: str | Path):
    """Yield every ``SE`` block of a plain-text ``.sim`` file as a :class:`SimEvent`."""
    event: SimEvent | None = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "SE" or line == "EN":
                if event is not None:
                    yield event
                event = None
                continue
            if line.startswith("ID "):
                event = SimEvent(int(line.split()[1]))
            elif event is not None and line.startswith("IA "):
                event._ias.append(_parse_ia_line(line))
            elif event is not None and line.startswith("HTsim "):
                event._hits.append(_parse_ht_line(line))
    if event is not None:
        yield event


def write_sim_file(path: str | Path, events: list[SimEvent]) -> Path:
    """Serialize events to cosima-style ``.sim`` text that :func:`iter_sim_events` parses back.

    Only the fields the pipeline consumes are written (event id, IA process/id/
    origin, hit position/energy/origins), so the output is a valid fixture for
    the fake reader, not for real MEGAlib.

    Args:
        path: Destination ``.sim`` path.
        events: The events to serialize, in file order.

    Returns:
        The written path.
    """
    lines = ["Type SIM", "Version 101", ""]
    for event in events:
        lines += ["SE", f"ID {event.GetID()} {event.GetID()}"]
        for i in range(event.GetNIAs()):
            ia = event.GetIAAt(i)
            lines.append(f"IA {ia.GetProcess()} {ia.GetId()};{ia.GetOriginID()};3;0;0;0;0")
        for i in range(event.GetNHTs()):
            ht = event.GetHTAt(i)
            pos = ht.GetPosition()
            origins = "".join(f";{o}" for o in ht._origins)
            lines.append(f"HTsim 3;{pos.X()};{pos.Y()};{pos.Z()};{ht.GetEnergy()};0{origins}")
    lines += ["EN", ""]
    path = Path(path)
    path.write_text("\n".join(lines))
    return path


class MFileEventsSim:
    """Fake ``.sim`` reader: streams :class:`SimEvent` objects from the text file."""

    def __init__(self, geometry: MDGeometryQuest | None = None):
        self._geometry = geometry
        self._events = None

    def Open(self, path) -> bool:
        path = Path(os.path.expandvars(str(path)))
        if not path.is_file():
            return False
        self._events = iter_sim_events(path)
        return True

    def GetNextEvent(self) -> SimEvent | None:
        if self._events is None:
            return None
        return next(self._events, None)

    def Close(self) -> None:
        self._events = None


############################################
# .tra side + toy backprojection
############################################


class MCoordinateSystem:
    c_Spheric = 0
    c_Galactic = 1


class MPhysicalEvent:
    c_Compton = 1
    c_Photo = 2
    c_Unidentifiable = 3

    _TYPES: ClassVar[dict[str, int]] = {"CO": c_Compton, "PH": c_Photo}

    def __init__(
        self, event_id: int, event_type: int, direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ):
        self._id = int(event_id)
        self._type = int(event_type)
        norm = math.sqrt(sum(c * c for c in direction))
        self.direction = tuple(c / norm for c in direction) if norm > 0 else (0.0, 0.0, 1.0)

    def GetId(self) -> int:
        return self._id

    def GetType(self) -> int:
        return self._type


# Synthetic-event alias mirroring SimEvent's builder role on the .sim side.
PhysicalEvent = MPhysicalEvent


def iter_tra_events(path: str | Path):
    """Yield every ``SE`` block of a ``.tra`` file as an :class:`MPhysicalEvent`.

    Only what the pipeline consumes is parsed: the event type (``ET``), the id
    (``ID``), and — for Compton events — a deterministic direction proxy taken
    from the first interaction position of the ``CD`` line (the cone apex), which
    the toy :func:`CallBackproject` paints at.
    """
    fields: dict[str, str] = {}

    def flush():
        if "ID" not in fields or "ET" not in fields:
            return None
        etype = MPhysicalEvent._TYPES.get(fields["ET"], MPhysicalEvent.c_Unidentifiable)
        direction = (0.0, 0.0, 1.0)
        if "CD" in fields:
            x, y, z = (float(v) for v in fields["CD"].split()[:3])
            if x or y or z:
                direction = (x, y, z)
        return MPhysicalEvent(int(fields["ID"].split()[0]), etype, direction)

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "SE":
                event = flush()
                fields = {}
                if event is not None:
                    yield event
                continue
            key, _, value = line.partition(" ")
            if key in ("ET", "ID", "CD"):
                fields[key] = value.strip()
    event = flush()
    if event is not None:
        yield event


def write_tra_file(path: str | Path, events: list[MPhysicalEvent]) -> Path:
    """Serialize events to revan-style ``.tra`` text that :func:`iter_tra_events` parses back.

    Compton events are written as ``ET CO`` with their direction on the ``CD``
    line; every other type collapses to ``ET PH`` (the distinction the pipeline
    consumes is Compton vs not).

    Args:
        path: Destination ``.tra`` path.
        events: The events to serialize, in file order.

    Returns:
        The written path.
    """
    lines = ["Type tra", ""]
    for event in events:
        compton = event.GetType() == MPhysicalEvent.c_Compton
        lines += ["SE", f"ET {'CO' if compton else 'PH'}", f"ID {event.GetId()}"]
        if compton:
            x, y, z = event.direction
            lines.append(f"CD {x} {y} {z}   0 0 0")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


class MFileEventsTra:
    """Fake ``.tra`` reader: streams :class:`MPhysicalEvent` objects."""

    def __init__(self) -> None:
        self._events = None

    def Open(self, path) -> bool:
        path = Path(os.path.expandvars(str(path)))
        if not path.is_file():
            return False
        self._events = iter_tra_events(path)
        return True

    def GetNextEvent(self) -> MPhysicalEvent | None:
        if self._events is None:
            return None
        return next(self._events, None)

    def Close(self) -> None:
        self._events = None


class MResponseGaussianByUncertainties:
    pass


class MBackprojectionFarField:
    """Fake backprojector: records the sky grid; the math lives in :func:`CallBackproject`."""

    def __init__(self, coordinate_system: int):
        self.coordinate_system = coordinate_system
        self.geometry = None
        self.response = None
        self.dims: tuple | None = None

    def SetGeometry(self, geometry) -> None:
        self.geometry = geometry

    def SetResponse(self, response) -> None:
        self.response = response

    def SetDimensions(
        self, phi_min, phi_max, n_phi, theta_min, theta_max, n_theta, r_min, r_max, n_r
    ) -> bool:
        self.dims = (
            float(phi_min),
            float(phi_max),
            int(n_phi),
            float(theta_min),
            float(theta_max),
            int(n_theta),
        )
        return True

    def PrepareBackprojection(self) -> None:
        return None


class _BackprojectResult:
    def __init__(self, success: bool, n_used: int, maximum: float):
        self.success = success
        self.n_used = n_used
        self.maximum = maximum


def direction_to_pixel(bp: MBackprojectionFarField, direction: tuple[float, float, float]) -> int:
    """Flat sky-pixel index of a unit direction on ``bp``'s grid (phi-major rows)."""
    phi_min, phi_max, n_phi, theta_min, theta_max, n_theta = bp.dims
    x, y, z = direction
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x) % (2.0 * math.pi)
    phi_idx = min(int((phi - phi_min) / (phi_max - phi_min) * n_phi), n_phi - 1)
    theta_idx = min(int((theta - theta_min) / (theta_max - theta_min) * n_theta), n_theta - 1)
    return theta_idx * n_phi + max(phi_idx, 0)


def CallBackproject(bp: MBackprojectionFarField, event: MPhysicalEvent, image, bins) -> _BackprojectResult:
    """Toy backprojection: one unit-weight pixel at the event's carried direction.

    Deterministic per event and grid — a test double for the imager's plumbing,
    not a model of MEGAlib's cone response. Non-Compton events never reach here
    (``project_event`` filters on ``GetType``), but fail them anyway for safety.
    """
    if event.GetType() != MPhysicalEvent.c_Compton or bp.dims is None:
        return _BackprojectResult(False, 0, 0.0)
    pixel = direction_to_pixel(bp, event.direction)
    image[0] = 1.0
    bins[0] = pixel
    return _BackprojectResult(True, 1, 1.0)
