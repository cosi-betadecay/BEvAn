import os

import ROOT as M

from utils.megalib_types import MDGeometryQuest, MFileEventsSim


class ChunkScopedIds:
    """Disambiguate event IDs from an mcosima-concatenated file.

    Every chunk of an mcosima concatenation restarts its event IDs at 1, so a raw
    ID names up to one event *per chunk*, not per file. Both the ``.sim`` and the
    ``.tra`` revan derived from it visit the chunks in the same order with strictly
    increasing IDs inside each chunk, so a downward (or repeated) ID step marks a
    chunk boundary. Feeding every event's ID *in file order* to :meth:`key` yields
    a ``(chunk, id)`` pair that is unique across the file and identical between the
    ``.sim`` and its ``.tra`` — the join key for matching events across the two.
    """

    def __init__(self) -> None:
        """Start at chunk 0 with no event seen yet."""
        self._chunk = 0
        self._prev: int | None = None

    def key(self, event_id: int) -> tuple[int, int]:
        """Chunk-scoped key for the next event in file order.

        Must be called for every event as it is read — skipping events would miss
        the ID reset that marks a chunk boundary.

        Args:
            event_id: The event's raw ID as stored in the file.

        Returns:
            ``(chunk, event_id)``, unique across the concatenated file.
        """
        if self._prev is not None and event_id <= self._prev:
            self._chunk += 1
        self._prev = event_id
        return (self._chunk, event_id)


def load_geometry(geometry_file: str) -> MDGeometryQuest:
    """Load libMEGAlib and scan a geometry setup file once.

    The returned geometry can be shared across many readers (see
    :func:`open_sim_reader`), so callers streaming several ``.sim`` files
    against one geometry avoid re-scanning it per file.

    Args:
        geometry_file: Path to the MEGAlib geometry setup file.

    Returns:
        The scanned ``MDGeometryQuest``.

    Raises:
        RuntimeError: If the geometry file cannot be loaded.
    """
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    m_global = M.MGlobal()
    m_global.Initialize()

    # Expand $MEGALIB / ${MEGALIB} so the path resolves regardless of CLI quoting.
    geometry_file = os.path.expandvars(geometry_file)

    geometry = M.MDGeometryQuest()
    if not geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")
    return geometry


def open_sim_reader(geometry: MDGeometryQuest, sim_file: str) -> MFileEventsSim:
    """Open a ``.sim`` file against an already-scanned geometry.

    Args:
        geometry: A geometry from :func:`load_geometry`.
        sim_file: Path to the ``.sim`` simulation file.

    Returns:
        An opened ``MFileEventsSim`` positioned at the first event; the caller
        owns it and should ``Close()`` it when done.

    Raises:
        RuntimeError: If the simulation file cannot be opened.
    """
    reader = M.MFileEventsSim(geometry)
    if not reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")
    return reader
