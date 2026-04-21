from typing import Any

import ROOT as M


def get_reader(geometry_file: str, sim_file: str) -> Any:
    """Initialize a MEGAlib simulation file reader with the specified geometry.

    Args:
        geometry_file (str): Path to the MEGAlib geometry setup file.
        sim_file (str): Path to the MEGAlib simulation file.

    Raises:
        RuntimeError: If the geometry file cannot be loaded.
        RuntimeError: If the simulation file cannot be opened.

    Returns:
        Any: A MEGAlib simulation reader initialized with the
        given geometry and simulation file.
    """
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    G = M.MGlobal()
    G.Initialize()

    Geometry = M.MDGeometryQuest()
    if not Geometry.ScanSetupFile(M.MString(geometry_file)):
        raise RuntimeError(f"Could not load geometry {geometry_file}")

    Reader = M.MFileEventsSim(Geometry)
    if not Reader.Open(M.MString(sim_file)):
        raise RuntimeError(f"Could not open simulation file {sim_file}")

    return Reader
