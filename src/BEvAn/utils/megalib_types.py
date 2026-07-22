from typing import Any, Protocol


class MDGeometryQuest(Protocol):
    """A scanned MEGAlib detector geometry (PyROOT ``M.MDGeometryQuest``).

    Built once from a ``.geo.setup`` file via ``ScanSetupFile``; readers are
    constructed against it, so it can be shared across many ``.sim`` files.
    """


class MSimEvent(Protocol):
    """One simulated event (PyROOT ``M.MSimEvent``).

    Holds the Monte-Carlo truth interactions (IAs) and the measured detector
    hits (HTs). Key accessors: ``GetNIAs()`` / ``GetIAAt(i)`` for truth records
    (e.g. ``ANNI`` annihilation vertices) and ``GetNHTs()`` / ``GetHTAt(i)`` for
    hits (position, energy, origin IA ids).
    """

    def GetNIAs(self) -> int:
        """Number of Monte-Carlo truth interactions (IAs) in the event."""

    def GetIAAt(self, i: int) -> Any:
        """The ``i``-th truth interaction (a dynamic PyROOT ``M.MSimIA`` proxy)."""

    def GetNHTs(self) -> int:
        """Number of detector hits (HTs) in the event."""

    def GetHTAt(self, i: int) -> Any:
        """The ``i``-th detector hit (a dynamic PyROOT ``M.MSimHT`` proxy)."""


class MFileEventsSim(Protocol):
    """MEGAlib ``.sim`` file reader (PyROOT ``M.MFileEventsSim``).

    Opens a simulation file written by cosima and streams its events.
    ``GetNextEvent()`` returns the next :class:`MSimEvent` (``None`` at EOF).
    """

    def GetNextEvent(self) -> MSimEvent | None:
        """The next event in the file, or ``None`` at end of file."""

    def Close(self) -> None:
        """Close the underlying file."""


class MPhysicalEvent(Protocol):
    """A reconstructed physical event from a ``.tra`` file (PyROOT ``M.MPhysicalEvent``).

    Produced by revan. ``GetType()`` identifies the event class; ``== c_Compton``
    selects Compton events used for far-field backprojection.
    """

    def GetType(self) -> int:
        """The event-class id (compared against ``M.MPhysicalEvent.c_Compton``)."""

    def GetId(self) -> int:
        """The event id assigned by revan."""
