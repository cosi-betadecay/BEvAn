from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import ROOT as M
import torch
from tqdm import tqdm

from utils.megalib_types import MPhysicalEvent
from utils.reader_extraction import ChunkScopedIds

############################################
# MEGAlib + wrapper setup (idempotent)
############################################


_MGLOBAL_INITIALIZED = False


def ensure_megalib_loaded() -> None:
    """Load libMEGAlib into the ROOT interpreter. No-op if already loaded."""
    global _MGLOBAL_INITIALIZED
    if not hasattr(M, "MBackprojectionFarField"):
        M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
    if not hasattr(M, "MBackprojectionFarField"):
        raise RuntimeError(
            "MEGAlib symbols not available. Source $MEGALIB/bin/source-megalib.sh before running Python."
        )
    if not _MGLOBAL_INITIALIZED:
        m_global = M.MGlobal()
        m_global.Initialize()
        _MGLOBAL_INITIALIZED = True


_WRAPPER_DECLARED = False


def declare_wrapper() -> None:
    """Inline a C++ helper so we can call Backproject cleanly from Python.

    The native signature is::

        bool Backproject(MPhysicalEvent*, double*, int*, int&, double&)

    PyROOT's handling of the ``int&`` / ``double&`` output refs is version-dependent.
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


############################################
# Imager
############################################


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
        phi_range: tuple[float, float] = (0.0, 2.0 * math.pi),
        theta_range: tuple[float, float] = (0.0, math.pi),
    ) -> None:
        """Build the backprojector, sky grid, and per-event scratch buffers.

        Args:
            geometry_file: MEGAlib geometry setup file (required by the reader).
            n_phi: Number of azimuthal (phi) sky bins.
            n_theta: Number of polar (theta) sky bins.
            coordinate_system: ``"spheric"`` or ``"galactic"``.
            phi_range: ``(min, max)`` phi extent in radians.
            theta_range: ``(min, max)`` theta extent in radians.

        Raises:
            ValueError: If ``coordinate_system`` is not recognized.
            RuntimeError: If the geometry fails to load or grid setup fails.
        """
        ensure_megalib_loaded()
        declare_wrapper()

        self.n_phi = int(n_phi)
        self.n_theta = int(n_theta)
        self.n_pixels = self.n_phi * self.n_theta
        self.phi_range = phi_range
        self.theta_range = theta_range

        # Geometry (required by the reader; optional for imaging itself).
        # Expand $MEGALIB / ${MEGALIB} here so the path resolves no matter how it
        # was quoted on the command line ($(MEGALIB) is left for ROOT to expand).
        geometry_file = os.path.expandvars(str(geometry_file))
        self.geometry = M.MDGeometryQuest()
        if not self.geometry.ScanSetupFile(M.MString(geometry_file)):
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

        # Accumulator across events + per-event scratch buffers.
        # _event_image/_event_bins must remain numpy: PyROOT binds them to the
        # C++ ``double*`` / ``int*`` arguments of CallBackproject via numpy's
        # array interface; torch tensors do not expose the same buffer.
        self.accumulated = torch.zeros(self.n_pixels, dtype=torch.float64)
        self._event_image = np.zeros(self.n_pixels, dtype=np.float64)
        self._event_bins = np.zeros(self.n_pixels, dtype=np.int32)

    def image_2d(self) -> torch.Tensor:
        """Accumulated image reshaped as ``(n_theta, n_phi)``."""
        return self.accumulated.reshape(self.n_theta, self.n_phi)

    def project_event(self, event: MPhysicalEvent) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Backproject one event into its ``(bins, values)`` cone contribution.

        The returned ``values`` tensor views the per-event scratch buffer, so it
        must be consumed (e.g. scatter-added) before the next projection overwrites it.

        Args:
            event: A reconstructed physical event from the ``.tra`` reader.

        Returns:
            The used pixel indices and their image values, or None if the event
            is not a Compton event or Backproject reported failure.
        """
        if event.GetType() != M.MPhysicalEvent.c_Compton:
            return None

        self._event_image.fill(0.0)
        self._event_bins.fill(0)

        result = M.CallBackproject(self.bp, event, self._event_image, self._event_bins)
        n = int(result.n_used)
        if not result.success or n == 0:
            return None

        bins = torch.from_numpy(self._event_bins[:n].astype(np.int64))
        values = torch.from_numpy(self._event_image[:n])
        return bins, values

    def _iter_tra(
        self, tra_file: str | Path, desc: str, total: int | None = None
    ) -> Iterator[MPhysicalEvent]:
        """Yield every event of ``tra_file`` in file order, with a progress bar.

        Args:
            tra_file: Path to a MEGAlib ``.tra`` file of reconstructed events.
            desc: Progress-bar description.
            total: Optional progress-bar total.

        Yields:
            Each physical event, ownership transferred to Python.

        Raises:
            RuntimeError: If the file cannot be opened.
        """
        reader = M.MFileEventsTra()
        if not reader.Open(M.MString(str(tra_file))):
            raise RuntimeError(f"Failed to open {tra_file}")
        try:
            for evt in tqdm(
                iter(lambda: reader.GetNextEvent(), None), desc=desc, unit=" events", total=total
            ):
                M.SetOwnership(evt, True)
                yield evt
        finally:
            reader.Close()

    def backproject_event(self, event: MPhysicalEvent) -> bool:
        """Add one event's cone contribution to the accumulator.

        Args:
            event: A reconstructed physical event from the ``.tra`` reader.

        Returns:
            bool: True on success, False if the event was skipped (non-Compton,
            or Backproject reported failure).
        """
        projected = self.project_event(event)
        if projected is None:
            return False

        # Scatter-add: each used bin receives its event image value.
        bins, values = projected
        self.accumulated.index_put_((bins,), values, accumulate=True)
        return True

    def backproject_file(
        self,
        tra_file: str | Path,
        max_events: int | None = None,
    ) -> int:
        """Read ``tra_file`` and backproject every Compton event in it.

        Args:
            tra_file: Path to a MEGAlib ``.tra`` file of reconstructed events.
            max_events: Optional cap on the number of events to accumulate.

        Returns:
            int: The number of events successfully accumulated.
        """
        count = 0
        for evt in self._iter_tra(tra_file, desc="Backprojecting events", total=max_events):
            if self.backproject_event(evt):
                count += 1
                if max_events is not None and count >= max_events:
                    break
        return count

    def backproject_file_grouped(
        self,
        tra_file: str | Path,
        groups: Mapping[str, set[tuple[int, int]]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Accumulate a separate sky image per event-ID group, in one pass over ``tra_file``.

        Events are matched by their chunk-scoped ``(chunk, id)`` key (see
        :class:`~utils.reader_extraction.ChunkScopedIds`) against each group's key
        set — e.g. the classifier-tagged sets from
        :meth:`~pipeline.eval.Evaluator.classify_events` — and each matched event's
        cone is added to that group's image. Unmatched events are read past (their
        IDs still advance the chunk tracking). The instance's main accumulated
        image is left untouched.

        Args:
            tra_file: Path to the MEGAlib ``.tra`` file the groups' keys refer to.
            groups: Group name -> set of chunk-scoped event keys.

        Returns:
            ``(images, counts)`` — per-group ``(n_theta, n_phi)`` sky images and
            per-group counts of successfully accumulated events.

        Raises:
            RuntimeError: If the file cannot be opened, or no event matched any
                nonempty group (a broken ``.sim``/``.tra`` ID join).
        """
        flat = {name: torch.zeros(self.n_pixels, dtype=torch.float64) for name in groups}
        counts = dict.fromkeys(groups, 0)
        ids = ChunkScopedIds()

        for evt in self._iter_tra(tra_file, desc="Backprojecting tagged events"):
            key = ids.key(int(evt.GetId()))
            matched = [name for name, keys in groups.items() if key in keys]
            if not matched:
                continue
            projected = self.project_event(evt)
            if projected is None:
                continue
            bins, values = projected
            for name in matched:
                flat[name].index_put_((bins,), values, accumulate=True)
                counts[name] += 1

        if any(groups.values()) and sum(counts.values()) == 0:
            raise RuntimeError(
                f"No event in {tra_file} matched any group: the .sim/.tra chunk-scoped ID join is broken."
            )
        images = {name: img.reshape(self.n_theta, self.n_phi) for name, img in flat.items()}
        return images, counts

    def peak_direction(self) -> tuple[float, float]:
        """``(theta, phi)`` of the brightest pixel, in radians."""
        img = self.image_2d()
        flat_idx = int(torch.argmax(img).item())
        theta_idx = flat_idx // self.n_phi
        phi_idx = flat_idx % self.n_phi

        theta_min, theta_max = self.theta_range
        phi_min, phi_max = self.phi_range
        theta = theta_min + (theta_idx + 0.5) * (theta_max - theta_min) / self.n_theta
        phi = phi_min + (phi_idx + 0.5) * (phi_max - phi_min) / self.n_phi
        return theta, phi

    def peak_direction_cartesian(self) -> torch.Tensor:
        """Unit vector from the coordinate origin to the peak pixel."""
        theta, phi = self.peak_direction()
        return torch.tensor(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ],
            dtype=torch.float64,
        )
