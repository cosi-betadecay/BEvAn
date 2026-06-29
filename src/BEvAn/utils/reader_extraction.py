import os

import ROOT as M

from utils.megalib_types import MFileEventsSim


def get_reader(geometry_file: str, sim_file: str) -> MFileEventsSim:
    """Open a MEGAlib ``.sim`` file and return its initialized event reader.

    Loads libMEGAlib, scans the geometry setup, and opens the simulation file.

    Args:
        geometry_file: Path to the MEGAlib geometry setup file.
        sim_file: Path to the ``.sim`` simulation file.

    Returns:
        An opened ``MFileEventsSim`` positioned at the first event.

    Raises:
        RuntimeError: If the geometry or simulation file cannot be loaded.
    """
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    m_global = M.MGlobal()
    m_global.Initialize()

    # Expand $MEGALIB / ${MEGALIB} so the path resolves regardless of CLI quoting.
    geometry_file = os.path.expandvars(geometry_file)

    geometry = M.MDGeometryQuest()
    if not geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")

    reader = M.MFileEventsSim(geometry)
    if not reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")

    return reader
