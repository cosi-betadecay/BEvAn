class MFileEventsSim:
    """MEGAlib ``.sim`` file reader (PyROOT ``M.MFileEventsSim``).

    Opens a simulation file written by cosima and streams its events.
    ``GetNextEvent()`` returns the next :class:`MSimEvent` (``None`` at EOF).
    """


class MSimEvent:
    """One simulated event (PyROOT ``M.MSimEvent``).

    Holds the Monte-Carlo truth interactions (IAs) and the measured detector
    hits (HTs). Key accessors: ``GetNIAs()`` / ``GetIAAt(i)`` for truth records
    (e.g. ``ANNI`` annihilation vertices) and ``GetNHTs()`` / ``GetHTAt(i)`` for
    hits (position, energy, origin IA ids).
    """


class MPhysicalEvent:
    """A reconstructed physical event from a ``.tra`` file (PyROOT ``M.MPhysicalEvent``).

    Produced by revan. ``GetType()`` identifies the event class; ``== c_Compton``
    selects Compton events used for far-field backprojection.
    """
