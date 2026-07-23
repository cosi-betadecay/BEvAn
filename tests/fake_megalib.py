from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np


class MString(str):
    """MEGAlib string: plain ``str`` semantics are all the pipeline uses."""

    __slots__ = ()


def SetOwnership(obj: object, flag: bool) -> None:
    """Stand in for PyROOT's ownership helper, which has nothing to manage here.

    The pipeline calls this on every event it takes from a reader; the fake keeps
    the call site unchanged but does no bookkeeping, since Python garbage
    collection already owns these objects.

    Args:
        obj: The object whose ownership the real PyROOT would transfer.
        flag: Whether Python should take ownership (ignored by the fake).
    """
    return None


class _GSystem:
    """Fake ``ROOT.gSystem``: accepts shared-library loads without doing anything."""

    def Load(self, path: str) -> int:
        """Pretend to load a shared library and report success.

        Args:
            path: The library path the pipeline asks ROOT to load.

        Returns:
            ``0``, the ROOT success code.
        """
        return 0


class _GInterpreter:
    """Fake ``ROOT.gInterpreter``: accepts C++ declarations without compiling them."""

    def Declare(self, code: str) -> bool:
        """Pretend to declare C++ code into the interpreter.

        Args:
            code: The C++ source the pipeline asks ROOT to declare.

        Returns:
            ``True``, signalling a successful declaration.
        """
        return True


gSystem = _GSystem()
gInterpreter = _GInterpreter()


class MGlobal:
    """Fake ``MGlobal``: MEGAlib's global-initialization entry point."""

    def Initialize(self) -> bool:
        """Pretend to initialize MEGAlib's globals.

        Returns:
            ``True``, signalling a successful initialization.
        """
        return True


class MDGeometryQuest:
    """Fake geometry: records the scanned path and 'scans' successfully iff it exists.

    The pipeline only needs the geometry to accept a setup file and report
    whether it was usable, so the fake reduces a scan to a filesystem existence
    check (after expanding environment variables the way MEGAlib does).

    Attributes:
        path: The most recently scanned setup-file path, or ``None`` before any scan.
    """

    def __init__(self) -> None:
        self.path: str | None = None

    def ScanSetupFile(self, path: str | Path) -> bool:
        """Record a geometry setup file and report whether it exists on disk.

        Args:
            path: The ``.geo.setup`` path to scan.

        Returns:
            ``True`` if the (variable-expanded) path is an existing file.
        """
        self.path = str(path)
        return Path(os.path.expandvars(self.path)).is_file()


############################################
# .sim side
############################################


class MVector:
    """Fake ``MVector``: a 3D position with MEGAlib's ``X``/``Y``/``Z`` accessors.

    Args:
        x: Cartesian x coordinate.
        y: Cartesian y coordinate.
        z: Cartesian z coordinate.
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        self._x, self._y, self._z = float(x), float(y), float(z)

    def X(self) -> float:
        """Return the x coordinate."""
        return self._x

    def Y(self) -> float:
        """Return the y coordinate."""
        return self._y

    def Z(self) -> float:
        """Return the z coordinate."""
        return self._z


class SimHit:
    """One ``HTsim`` line: a detector hit's position, energy, and origin IA ids.

    Args:
        position: The hit's ``(x, y, z)`` position.
        energy: The energy deposited in the hit, in keV.
        origins: The ids of the interactions this hit is a descendant of, used by
            ``IsOrigin`` for truth labelling.
    """

    def __init__(
        self, position: tuple[float, float, float], energy: float, origins: tuple[int, ...] = ()
    ) -> None:
        self._position = MVector(*position)
        self._energy = float(energy)
        self._origins = tuple(int(o) for o in origins)

    def GetPosition(self) -> MVector:
        """Return the hit position as an :class:`MVector`."""
        return self._position

    def GetEnergy(self) -> float:
        """Return the deposited energy in keV."""
        return self._energy

    def IsOrigin(self, ia_id: int) -> bool:
        """Report whether this hit descends from the given interaction id.

        Args:
            ia_id: The interaction id to test against the hit's origin chain.

        Returns:
            ``True`` if ``ia_id`` is one of the hit's recorded origins.
        """
        return int(ia_id) in self._origins


class SimIA:
    """One ``IA`` line: an interaction's process name, id, and originating id.

    Args:
        process: The interaction process name (e.g. ``"ANNI"``, ``"COMP"``).
        ia_id: The interaction's positional id within the event.
        origin_id: The id of the interaction that spawned this one.
    """

    def __init__(self, process: str, ia_id: int, origin_id: int) -> None:
        self._process = MString(process)
        self._id = int(ia_id)
        self._origin_id = int(origin_id)

    def GetProcess(self) -> MString:
        """Return the interaction process name."""
        return self._process

    def GetId(self) -> int:
        """Return the interaction's positional id."""
        return self._id

    def GetOriginID(self) -> int:
        """Return the id of the interaction that spawned this one."""
        return self._origin_id


class SimEvent:
    """One ``SE`` block of a ``.sim`` file (also a synthetic-event builder).

    Doubles as the builder the fixtures use to assemble synthetic events in
    memory before serializing them, mirroring what the reader yields on parse.

    Args:
        event_id: The event's id.
        hits: The event's hits, or ``None`` for a hitless event.
        ias: The event's interactions, or ``None`` for none.
    """

    def __init__(
        self, event_id: int, hits: list[SimHit] | None = None, ias: list[SimIA] | None = None
    ) -> None:
        self._id = int(event_id)
        self._hits = list(hits or [])
        self._ias = list(ias or [])

    def GetID(self) -> int:
        """Return the event id."""
        return self._id

    def GetNHTs(self) -> int:
        """Return the number of hits in the event."""
        return len(self._hits)

    def GetHTAt(self, i: int) -> SimHit:
        """Return the hit at index ``i``.

        Args:
            i: Zero-based hit index.

        Returns:
            The hit at that index.
        """
        return self._hits[i]

    def GetNIAs(self) -> int:
        """Return the number of interactions in the event."""
        return len(self._ias)

    def GetIAAt(self, i: int) -> SimIA:
        """Return the interaction at index ``i``.

        Args:
            i: Zero-based interaction index.

        Returns:
            The interaction at that index.
        """
        return self._ias[i]


def _parse_ia_line(line: str) -> SimIA:
    """Parse one ``IA`` line into a :class:`SimIA`.

    Reads the ``"IA COMP 12;10;3;<time>;<x>;<y>;<z>;..."`` layout down to the
    fields the pipeline consumes: process name, interaction id, and origin id.

    Args:
        line: The raw ``IA`` line, including its ``"IA "`` prefix.

    Returns:
        The parsed interaction.
    """
    head, *rest = line[3:].split(";")
    process, ia_id = head.split()
    return SimIA(process, int(ia_id), int(rest[0]))


def _parse_ht_line(line: str) -> SimHit:
    """Parse one ``HTsim`` line into a :class:`SimHit`.

    Reads the ``"HTsim 3; x; y; z; energy; time; origin;origin;..."`` layout;
    trailing origin ids are optional.

    Args:
        line: The raw ``HTsim`` line, including its ``"HTsim "`` prefix.

    Returns:
        The parsed hit.
    """
    parts = line[6:].split(";")
    x, y, z, energy = (float(p) for p in parts[1:5])
    origins = tuple(int(p) for p in parts[6:] if p.strip())
    return SimHit((x, y, z), energy, origins)


def iter_sim_events(path: str | Path) -> Iterator[SimEvent]:
    """Yield every ``SE`` block of a plain-text ``.sim`` file as a :class:`SimEvent`.

    Args:
        path: The ``.sim`` file to stream.

    Yields:
        Each event in file order.
    """
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
    """Fake ``.sim`` reader: streams :class:`SimEvent` objects from the text file.

    Mirrors MEGAlib's open/next/close reader protocol over the plain-text parser
    so the pipeline's extraction code runs unchanged.

    Args:
        geometry: The geometry the real reader would associate with the file;
            accepted but unused by the fake.
    """

    def __init__(self, geometry: MDGeometryQuest | None = None) -> None:
        self._geometry = geometry
        self._events: Iterator[SimEvent] | None = None

    def Open(self, path: str | Path) -> bool:
        """Open a ``.sim`` file and begin streaming its events.

        Args:
            path: The ``.sim`` path to open (environment variables are expanded).

        Returns:
            ``True`` if the file exists and was opened, ``False`` otherwise.
        """
        path = Path(os.path.expandvars(str(path)))
        if not path.is_file():
            return False
        self._events = iter_sim_events(path)
        return True

    def GetNextEvent(self) -> SimEvent | None:
        """Return the next event, or ``None`` once the stream is exhausted."""
        if self._events is None:
            return None
        return next(self._events, None)

    def Close(self) -> None:
        """Release the underlying event stream."""
        self._events = None


############################################
# .tra side + toy backprojection
############################################


class MCoordinateSystem:
    """Fake ``MCoordinateSystem``: the coordinate-system enum the imager selects.

    Attributes:
        c_Spheric: Spherical coordinate-system code.
        c_Galactic: Galactic coordinate-system code.
    """

    c_Spheric = 0
    c_Galactic = 1


class MPhysicalEvent:
    """One reconstructed ``.tra`` event: its id, type, and a unit direction.

    Also serves as the synthetic-event builder for the ``.tra`` side (aliased as
    :data:`PhysicalEvent`), mirroring :class:`SimEvent`'s role on the ``.sim``
    side. The direction is normalized on construction so Compton events always
    carry a unit vector for the toy backprojection.

    Args:
        event_id: The event's id.
        event_type: The event-type code (one of ``c_Compton``/``c_Photo``/
            ``c_Unidentifiable``).
        direction: The event direction; normalized to a unit vector, or replaced
            by ``(0, 0, 1)`` if it has zero length.

    Attributes:
        c_Compton: Compton-event type code.
        c_Photo: Photoelectric-event type code.
        c_Unidentifiable: Fallback type code for unparsed event types.
        direction: The event's unit direction vector.
    """

    c_Compton = 1
    c_Photo = 2
    c_Unidentifiable = 3

    _TYPES: ClassVar[dict[str, int]] = {"CO": c_Compton, "PH": c_Photo}

    def __init__(
        self, event_id: int, event_type: int, direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> None:
        self._id = int(event_id)
        self._type = int(event_type)
        norm = math.sqrt(sum(c * c for c in direction))
        self.direction = tuple(c / norm for c in direction) if norm > 0 else (0.0, 0.0, 1.0)

    def GetId(self) -> int:
        """Return the event id."""
        return self._id

    def GetType(self) -> int:
        """Return the event-type code."""
        return self._type


# Synthetic-event alias mirroring SimEvent's builder role on the .sim side.
PhysicalEvent = MPhysicalEvent


def iter_tra_events(path: str | Path) -> Iterator[MPhysicalEvent]:
    """Yield every ``SE`` block of a ``.tra`` file as an :class:`MPhysicalEvent`.

    Only what the pipeline consumes is parsed: the event type (``ET``), the id
    (``ID``), and — for Compton events — a deterministic direction proxy taken
    from the first interaction position of the ``CD`` line (the cone apex), which
    the toy :func:`CallBackproject` paints at.

    Args:
        path: The ``.tra`` file to stream.

    Yields:
        Each parsed event in file order.
    """
    fields: dict[str, str] = {}

    def flush() -> MPhysicalEvent | None:
        """Build the pending event from accumulated fields, or ``None`` if incomplete."""
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
    """Fake ``.tra`` reader: streams :class:`MPhysicalEvent` objects.

    Mirrors MEGAlib's open/next/close reader protocol over the plain-text ``.tra``
    parser so the pipeline's imaging code runs unchanged.
    """

    def __init__(self) -> None:
        self._events: Iterator[MPhysicalEvent] | None = None

    def Open(self, path: str | Path) -> bool:
        """Open a ``.tra`` file and begin streaming its events.

        Args:
            path: The ``.tra`` path to open (environment variables are expanded).

        Returns:
            ``True`` if the file exists and was opened, ``False`` otherwise.
        """
        path = Path(os.path.expandvars(str(path)))
        if not path.is_file():
            return False
        self._events = iter_tra_events(path)
        return True

    def GetNextEvent(self) -> MPhysicalEvent | None:
        """Return the next event, or ``None`` once the stream is exhausted."""
        if self._events is None:
            return None
        return next(self._events, None)

    def Close(self) -> None:
        """Release the underlying event stream."""
        self._events = None


class MResponseGaussianByUncertainties:
    """Fake response: a placeholder the backprojector accepts but never queries."""


class MBackprojectionFarField:
    """Fake backprojector: records the sky grid; the math lives in :func:`CallBackproject`.

    Captures the geometry, response, and sky-grid dimensions the imager sets, so
    the toy backprojection can map an event direction to a flat pixel index.

    Args:
        coordinate_system: The coordinate-system code the imager selects.

    Attributes:
        coordinate_system: The selected coordinate-system code.
        geometry: The associated geometry, or ``None`` before ``SetGeometry``.
        response: The associated response, or ``None`` before ``SetResponse``.
        dims: The captured ``(phi_min, phi_max, n_phi, theta_min, theta_max,
            n_theta)`` grid, or ``None`` before ``SetDimensions``.
    """

    def __init__(self, coordinate_system: int) -> None:
        self.coordinate_system = coordinate_system
        self.geometry: MDGeometryQuest | None = None
        self.response: MResponseGaussianByUncertainties | None = None
        self.dims: tuple[float, float, int, float, float, int] | None = None

    def SetGeometry(self, geometry: MDGeometryQuest) -> None:
        """Associate a geometry with the backprojector.

        Args:
            geometry: The geometry to record.
        """
        self.geometry = geometry

    def SetResponse(self, response: MResponseGaussianByUncertainties) -> None:
        """Associate a response with the backprojector.

        Args:
            response: The response to record.
        """
        self.response = response

    def SetDimensions(
        self,
        phi_min: float,
        phi_max: float,
        n_phi: int,
        theta_min: float,
        theta_max: float,
        n_theta: int,
        r_min: float,
        r_max: float,
        n_r: int,
    ) -> bool:
        """Record the sky-grid dimensions used to bin backprojected directions.

        Only the phi/theta grid is retained; the radial arguments are accepted to
        match MEGAlib's signature but ignored by the far-field fake.

        Args:
            phi_min: Lower phi bound of the grid.
            phi_max: Upper phi bound of the grid.
            n_phi: Number of phi bins.
            theta_min: Lower theta bound of the grid.
            theta_max: Upper theta bound of the grid.
            n_theta: Number of theta bins.
            r_min: Lower radial bound (ignored).
            r_max: Upper radial bound (ignored).
            n_r: Number of radial bins (ignored).

        Returns:
            ``True``, signalling the dimensions were accepted.
        """
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
        """Pretend to prepare backprojection state; the fake needs none."""
        return None


class _BackprojectResult:
    """Outcome of one toy backprojection: success plus the pixels it touched.

    Args:
        success: Whether the event was backprojected.
        n_used: The number of sky pixels the event contributed to.
        maximum: The peak pixel weight the event deposited.

    Attributes:
        success: Whether the event was backprojected.
        n_used: The number of sky pixels the event contributed to.
        maximum: The peak pixel weight the event deposited.
    """

    def __init__(self, success: bool, n_used: int, maximum: float) -> None:
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


def CallBackproject(
    bp: MBackprojectionFarField, event: MPhysicalEvent, image: np.ndarray, bins: np.ndarray
) -> _BackprojectResult:
    """Toy backprojection: one unit-weight pixel at the event's carried direction.

    Deterministic per event and grid — a test double for the imager's plumbing,
    not a model of MEGAlib's cone response. Non-Compton events never reach here
    (``project_event`` filters on ``GetType``), but fail them anyway for safety.
    Writes the single touched pixel into ``image``/``bins`` in place, matching the
    real helper's ``double*``/``int*`` output arguments.

    Args:
        bp: The backprojector carrying the sky grid.
        event: The event to backproject.
        image: Output pixel-weight buffer; the touched pixel's weight is written
            to index 0.
        bins: Output pixel-index buffer; the touched pixel's index is written to
            index 0.

    Returns:
        The backprojection outcome (success flag, pixels used, peak weight).
    """
    if event.GetType() != MPhysicalEvent.c_Compton or bp.dims is None:
        return _BackprojectResult(False, 0, 0.0)
    pixel = direction_to_pixel(bp, event.direction)
    image[0] = 1.0
    bins[0] = pixel
    return _BackprojectResult(True, 1, 1.0)
