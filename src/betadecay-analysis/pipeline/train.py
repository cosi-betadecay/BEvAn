from pathlib import Path

import torch

from dataset.datasets import BUCKETS, FEATURES
from modeling.matrix_calculations import bins_from_counts, build_density_matrix, build_density_matrix_1d

# Bumped whenever the saved-artifact layout changes incompatibly; load refuses
# a payload whose version differs so a stale file fails loudly.
SAVE_FORMAT_VERSION = 1


class Trainer:
    """Train a per-bucket generative model for the β+/bg classification task.

    Each bucket has a different set of features (bucket 1: delta_E, bucket 2:
    delta_E & ARM, bucket 3: delta_E & ARM + delta_E & anni), so we build one
    independent model per bucket. Each model consists of a set of density terms
    (1D or 2D) and a class prior. The density terms are estimated from the
    training data using histogramming with Laplace smoothing. The class prior is
    estimated from the full per-bucket class counts.
    """

    def __init__(
        self,
        rho_floor: float | dict[int, float] = 10.0,
        joint_smoothing: float = 0.5,
        marginal_smoothing: float = 0.0,
    ) -> None:
        """Configure the bin-size rule and Laplace smoothing strengths.

        Args:
            rho_floor: Target events per histogram cell. Each density term sizes
                its bins-per-axis to the scarcer class via :func:`bins_from_counts`,
                so cells stay populated regardless of how many events a simulation
                yields; smaller ``rho_floor`` -> finer bins -> higher variance. A
                scalar applies to every bucket; a ``{bucket: rho_floor}`` dict sets
                each hit-multiplicity bucket independently (so the resolution can
                match a per-bucket bin budget like the hardcoded ``{1:25, 2:8, 3:35}``).
            joint_smoothing: Laplace pseudo-counts for the 2D joint densities.
            marginal_smoothing: Laplace pseudo-counts for the 1D delta_E marginal.
        """
        self.rho_floor = rho_floor
        self.joint_smoothing = joint_smoothing  # Laplace pseudo-counts for the 2D joints (sparse-bucket safe)
        self.marginal_smoothing = marginal_smoothing  # for the 1D delta_E marginal

    def config(self) -> dict:
        """The constructor kwargs that define this trainer's binning and smoothing.

        Re-passing the result to :class:`Trainer` reproduces an identically
        configured trainer, so it is what :meth:`save` records for :meth:`load`.

        Returns:
            ``{"rho_floor": ..., "joint_smoothing": ..., "marginal_smoothing": ...}``.
        """
        return {
            "rho_floor": self.rho_floor,
            "joint_smoothing": self.joint_smoothing,
            "marginal_smoothing": self.marginal_smoothing,
        }

    def fit(self, train: dict[str, dict[int, dict[str, torch.Tensor]]]) -> "Trainer":
        """Fit one independent density model per hit-multiplicity bucket.

        Args:
            train: Nested per-class, per-bucket training feature dict.

        Returns:
            ``self``, with ``models`` populated (one model per bucket).
        """
        # One independent model per hit-multiplicity bucket.
        self.models = {b: self.fit_bucket(train["bdecay"][b], train["bg"][b], b) for b in BUCKETS}
        return self

    def save(
        self,
        path: str | Path,
        threshold_offset: float = 0.0,
        provenance: dict | None = None,
    ) -> None:
        """Serialize the fitted per-bucket models to a self-describing torch artifact.

        Alongside the density matrices the payload records this trainer's config
        (so :meth:`load` can rebuild it), a feature/bucket signature validated on
        load, the calibrated threshold offset (already baked into each model's
        ``log_threshold``, so this copy is for traceability only), and free-form
        provenance. Every value is a plain scalar, string, or tensor, so the
        artifact reloads without re-fitting and without a live MEGAlib reader.

        Args:
            path: Destination ``.pt`` file; parent directories are created.
            threshold_offset: The global margin offset from calibration, recorded
                for traceability (its effect already lives in ``log_threshold``).
            provenance: Optional metadata, e.g. the source sim/tra/geometry paths.

        Raises:
            RuntimeError: If called before :meth:`fit` populated ``models``.
        """
        if not hasattr(self, "models"):
            raise RuntimeError("Trainer must be fit before it can be saved.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": SAVE_FORMAT_VERSION,
            "feature_signature": {"features": list(FEATURES), "buckets": list(BUCKETS)},
            "models": self.models,
            "trainer_kwargs": self.config(),
            "threshold_offset": threshold_offset,
            "provenance": provenance or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "Trainer":
        """Reconstruct a fitted trainer from a :meth:`save` artifact, without re-fitting.

        Validates the format version and feature/bucket signature so a stale
        artifact fails loudly instead of scoring against a mismatched feature
        layout, rebuilds the trainer from the stored config, and restores its
        per-bucket density models. The result is ready for
        :class:`~pipeline.eval.Evaluator` and reproduces the saved decision rule
        exactly (the calibrated ``log_threshold`` rides along inside ``models``).

        Args:
            path: Path to a ``.pt`` artifact written by :meth:`save`.

        Returns:
            A trainer with ``models`` populated, equivalent to the saved one.

        Raises:
            ValueError: If the artifact's format version or feature/bucket
                signature does not match this code.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("format_version")
        if version != SAVE_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format version {version!r}; expected {SAVE_FORMAT_VERSION}.")
        signature = payload.get("feature_signature")
        expected = {"features": list(FEATURES), "buckets": list(BUCKETS)}
        if signature != expected:
            raise ValueError(
                f"Model feature signature {signature} does not match this code {expected}; "
                "the artifact was trained against a different feature/bucket layout."
            )
        trainer = cls(**payload["trainer_kwargs"])
        trainer.models = payload["models"]
        return trainer

    def finite(self, *cols: torch.Tensor) -> torch.Tensor:
        """Boolean mask of positions that are finite across all given feature columns."""
        mask = torch.ones_like(cols[0], dtype=torch.bool)
        for c in cols:
            mask = mask & torch.isfinite(c)
        return mask

    def fit_bucket(self, bd: dict[str, torch.Tensor], bg: dict[str, torch.Tensor], bucket: int) -> dict:
        """Build the density terms and class prior for one bucket.

        Priors use the full per-bucket class counts (incl. NaN-feature events);
        densities are estimated from finite values only. Bucket 1 gets a 1D
        delta_E term, bucket 2 a 2D (delta_E, ARM) term, and bucket 3 adds a 2D
        (delta_E, anni) term. An empty class yields a prior-only model.

        Args:
            bd: β-decay per-feature tensors for this bucket.
            bg: Background per-feature tensors for this bucket.
            bucket: Hit-multiplicity bucket (1, 2, or 3).

        Returns:
            Dict with ``n_beta``, ``n_bg``, and ``terms``.
        """
        n_beta = int(bd["delta_E"].numel())
        n_bg = int(bg["delta_E"].numel())
        model = {"n_beta": n_beta, "n_bg": n_bg, "terms": []}
        if n_beta == 0 or n_bg == 0:
            return model  # prior-only: cannot estimate class-conditional densities

        rho = self.rho_floor[bucket] if isinstance(self.rho_floor, dict) else self.rho_floor
        if bucket == 1:
            self.add_1d(model, bd, bg, "delta_E", rho)
        elif bucket == 2:
            self.add_2d(model, bd, bg, "delta_E", "arm", "linear", None, rho)
        else:  # bucket 3
            self.add_2d(model, bd, bg, "delta_E", "arm", "linear", None, rho)
            self.add_2d(model, bd, bg, "delta_E", "anni", "linear", None, rho)
        return model

    def add_1d(
        self,
        model: dict,
        bd: dict[str, torch.Tensor],
        bg: dict[str, torch.Tensor],
        feat: str,
        rho_floor: float,
    ) -> None:
        """Append a 1D density term over ``feat`` to ``model`` (in place).

        Bins-per-axis is sized from the scarcer class's finite count via
        :func:`bins_from_counts` (floored at 5 bins). Per-class densities share
        the pooled-data edges and use the marginal smoothing. No-op unless both
        classes have a finite value (a ratio needs both densities).

        Args:
            model: Model dict whose ``terms`` list is appended to.
            bd: β-decay per-feature tensors.
            bg: Background per-feature tensors.
            feat: Feature name to histogram.
            rho_floor: Target events per cell for this bucket's bin rule.
        """
        fb = bd[feat][self.finite(bd[feat])]
        fg = bg[feat][self.finite(bg[feat])]
        if fb.numel() == 0 or fg.numel() == 0:
            return  # need both class densities -> prior-only
        n_bins = bins_from_counts(min(fb.numel(), fg.numel()), d=1, rho_floor=rho_floor, min_bins=5)
        _, x_bins = build_density_matrix_1d(
            torch.cat([fb, fg]), n_bins_x=n_bins, spacing_x="log", log_x_floor=1e-3
        )
        beta, _ = build_density_matrix_1d(fb, x_bins=x_bins, smoothing=self.marginal_smoothing)
        bgm, _ = build_density_matrix_1d(fg, x_bins=x_bins, smoothing=self.marginal_smoothing)
        model["terms"].append({"kind": "1d", "xfeat": feat, "x_bins": x_bins, "beta": beta, "bg": bgm})

    def add_2d(
        self,
        model: dict,
        bd: dict[str, torch.Tensor],
        bg: dict[str, torch.Tensor],
        xfeat: str,
        yfeat: str,
        spacing_y: str,
        floor_y: float | None,
        rho_floor: float,
    ) -> None:
        """Append a 2D density term over ``(xfeat, yfeat)`` to ``model`` (in place).

        Bins-per-axis is sized from the scarcer class's finite-pair count via
        :func:`bins_from_counts` (``d=2``, floored at 4 -> a 4x4 grid). Per-class
        densities share the pooled-data edges and use the joint smoothing. No-op
        unless both classes have a finite pair (a ratio needs both densities).

        Args:
            model: Model dict whose ``terms`` list is appended to.
            bd: β-decay per-feature tensors.
            bg: Background per-feature tensors.
            xfeat: First (x-axis) feature name; x is log-spaced.
            yfeat: Second (y-axis) feature name.
            spacing_y: Y-axis bin spacing, ``"log"`` or ``"linear"``.
            floor_y: Log-floor for the y-axis (ignored when linear).
            rho_floor: Target events per cell for this bucket's bin rule.
        """
        mb = self.finite(bd[xfeat], bd[yfeat])
        mg = self.finite(bg[xfeat], bg[yfeat])
        n_beta, n_bg = int(mb.sum()), int(mg.sum())
        if n_beta == 0 or n_bg == 0:
            return  # need both class densities -> prior-only
        n_bins = bins_from_counts(min(n_beta, n_bg), d=2, rho_floor=rho_floor, min_bins=4)
        _, x_bins, y_bins = build_density_matrix(
            torch.cat([bd[xfeat][mb], bg[xfeat][mg]]),
            torch.cat([bd[yfeat][mb], bg[yfeat][mg]]),
            spacing_x="log",
            spacing_y=spacing_y,
            log_x_floor=1e-3,
            log_y_floor=floor_y,
            n_bins_x=n_bins,
            n_bins_y=n_bins,
        )
        beta, _, _ = build_density_matrix(
            bd[xfeat][mb], bd[yfeat][mb], x_bins=x_bins, y_bins=y_bins, smoothing=self.joint_smoothing
        )
        bgm, _, _ = build_density_matrix(
            bg[xfeat][mg], bg[yfeat][mg], x_bins=x_bins, y_bins=y_bins, smoothing=self.joint_smoothing
        )
        model["terms"].append(
            {
                "kind": "2d",
                "xfeat": xfeat,
                "yfeat": yfeat,
                "x_bins": x_bins,
                "y_bins": y_bins,
                "beta": beta,
                "bg": bgm,
            }
        )
