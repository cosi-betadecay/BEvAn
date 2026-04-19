from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT as M
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# MEGAlib + wrapper setup (idempotent)
# --------------------------------------------------------------------------- #


_MGLOBAL_INITIALIZED = False


def _ensure_megalib_loaded() -> None:
    """Load libMEGAlib into the ROOT interpreter. No-op if already loaded."""
    global _MGLOBAL_INITIALIZED
    if not hasattr(M, "MBackprojectionFarField"):
        M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    if not hasattr(M, "MBackprojectionFarField"):
        raise RuntimeError(
            "MEGAlib symbols not available. Source $MEGALIB/bin/source-megalib.sh before running Python."
        )
    if not _MGLOBAL_INITIALIZED:
        G = M.MGlobal()
        G.Initialize()
        _MGLOBAL_INITIALIZED = True


_WRAPPER_DECLARED = False


def _declare_wrapper() -> None:
    """Inline a C++ helper so we can call Backproject cleanly from Python.

    The native signature is
        bool Backproject(MPhysicalEvent*, double*, int*, int&, double&)
    PyROOT's handling of the `int&` / `double&` output refs is version-dependent.
    Returning a POD struct from a C++ wrapper dodges that entirely.
    """
    global _WRAPPER_DECLARED
    if _WRAPPER_DECLARED:
        return

    M.gInterpreter.Declare(
        """
        struct BackprojectResult {
            bool success;
            int n_used;
            double maximum;
        };

        BackprojectResult CallBackproject(
            MBackprojectionFarField* bp,
            MPhysicalEvent* event,
            double* image,
            int* bins
        ) {
            BackprojectResult r;
            r.n_used = 0;
            r.maximum = 0.0;
            r.success = bp->Backproject(event, image, bins, r.n_used, r.maximum);
            return r;
        }
        """
    )
    _WRAPPER_DECLARED = True


# --------------------------------------------------------------------------- #
# Imager
# --------------------------------------------------------------------------- #


class FarFieldImager:
    """Far-field Compton backprojection for point-source localization.

    Thin wrapper over MEGAlib's MBackprojectionFarField. Maintains an
    accumulated sky image; reports the peak direction after events are added.
    """

    def __init__(
        self,
        geometry_file: str | Path,
        n_phi: int = 360,
        n_theta: int = 180,
        coordinate_system: str = "spheric",
        phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
        theta_range: tuple[float, float] = (0.0, np.pi),
    ) -> None:
        _ensure_megalib_loaded()
        _declare_wrapper()

        self.n_phi = int(n_phi)
        self.n_theta = int(n_theta)
        self.n_pixels = self.n_phi * self.n_theta
        self.phi_range = phi_range
        self.theta_range = theta_range

        # Geometry (required by the reader; optional for imaging itself)
        self.geometry = M.MDGeometryQuest()
        if not self.geometry.ScanSetupFile(M.MString(str(geometry_file))):
            raise RuntimeError(f"Failed to load geometry from {geometry_file}")

        # Coordinate system
        cs_map = {
            "spheric": M.MCoordinateSystem.c_Spheric,
            "galactic": M.MCoordinateSystem.c_Galactic,
        }
        if coordinate_system.lower() not in cs_map:
            raise ValueError(f"Unknown coordinate_system: {coordinate_system!r}")
        cs_enum = cs_map[coordinate_system.lower()]

        # Backprojector
        self.bp = M.MBackprojectionFarField(cs_enum)
        self.bp.SetGeometry(self.geometry)

        # Detector response — Gaussian using per-event uncertainties. The
        # simplest MEGAlib response that needs no training data or .rsp file.
        self.response = M.MResponseGaussianByUncertainties()
        self.bp.SetResponse(self.response)

        # Set up the sky grid (phi = x1, theta = x2; radius = 1 bin, ignored)
        ok = self.bp.SetDimensions(
            float(phi_range[0]),
            float(phi_range[1]),
            self.n_phi,
            float(theta_range[0]),
            float(theta_range[1]),
            self.n_theta,
            0.0,
            0.0,
            1,
        )
        if not ok:
            raise RuntimeError("MBackprojectionFarField.SetDimensions failed")
        self.bp.PrepareBackprojection()

        # Accumulator across events + per-event scratch buffers
        self.accumulated = np.zeros(self.n_pixels, dtype=np.float64)
        self._event_image = np.zeros(self.n_pixels, dtype=np.float64)
        self._event_bins = np.zeros(self.n_pixels, dtype=np.int32)

    # --- image management -------------------------------------------------- #

    def reset(self) -> None:
        """Zero the accumulated image."""
        self.accumulated.fill(0.0)

    def image_2d(self) -> np.ndarray:
        """Accumulated image reshaped as ``(n_theta, n_phi)``."""
        return self.accumulated.reshape(self.n_theta, self.n_phi)

    # --- event ingestion --------------------------------------------------- #

    def backproject_event(self, event) -> bool:
        """Add one event's cone contribution to the accumulator.

        Returns True on success, False if the event was skipped
        (non-Compton, or Backproject reported failure).
        """
        if event.GetType() != M.MPhysicalEvent.c_Compton:
            return False

        self._event_image.fill(0.0)
        self._event_bins.fill(0)

        result = M.CallBackproject(self.bp, event, self._event_image, self._event_bins)
        n = int(result.n_used)
        if not result.success or n == 0:
            return False

        # Scatter-add: each used bin receives its event image value.
        np.add.at(self.accumulated, self._event_bins[:n], self._event_image[:n])
        return True

    def backproject_file(
        self,
        tra_file: str | Path,
        max_events: int | None = None,
    ) -> int:
        """Read ``tra_file`` and backproject every Compton event in it.

        Returns the number of events successfully accumulated.
        """
        reader = M.MFileEventsTra()
        if not reader.Open(M.MString(str(tra_file))):
            raise RuntimeError(f"Failed to open {tra_file}")

        count = 0
        try:
            for evt in tqdm(
                iter(lambda: reader.GetNextEvent(), None),
                desc="Backprojecting events",
                unit=" events",
                total=max_events,
            ):
                M.SetOwnership(evt, True)
                if self.backproject_event(evt):
                    count += 1
                    if max_events is not None and count >= max_events:
                        break
        finally:
            reader.Close()
        return count

    # --- peak extraction --------------------------------------------------- #

    def peak_direction(self) -> tuple[float, float]:
        """``(theta, phi)`` of the brightest pixel, in radians."""
        img = self.image_2d()
        theta_idx, phi_idx = np.unravel_index(int(np.argmax(img)), img.shape)

        theta_min, theta_max = self.theta_range
        phi_min, phi_max = self.phi_range
        theta = theta_min + (theta_idx + 0.5) * (theta_max - theta_min) / self.n_theta
        phi = phi_min + (phi_idx + 0.5) * (phi_max - phi_min) / self.n_phi
        return theta, phi

    def peak_direction_cartesian(self) -> np.ndarray:
        """Unit vector from the coordinate origin to the peak pixel."""
        theta, phi = self.peak_direction()
        return np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

    # --- persistence ------------------------------------------------------- #

    def save_image_numpy(self, path: str | Path) -> None:
        """Save the accumulated image as ``.npy``."""
        np.save(str(path), self.image_2d())


if __name__ == "__main__":
    import os

    repo_root = Path(__file__).resolve().parents[3]
    geometry_file = os.path.expandvars("$MEGALIB/resource/examples/geomega/special/Max.geo.setup")
    tra_file = repo_root / "data" / "Activation.tra"

    imager = FarFieldImager(
        geometry_file=geometry_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(tra_file)
    print(f"Accumulated {n_events} Compton events from {tra_file}")

    theta_rad, phi_rad = imager.peak_direction()
    print(f"theta_rad = {theta_rad}, phi_rad = {phi_rad}")

    unit_vector = imager.peak_direction_cartesian()
    print(f"unit vector = {unit_vector}")
