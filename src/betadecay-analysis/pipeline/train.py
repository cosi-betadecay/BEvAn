import torch
from dataset.datasets import BUCKETS
from modeling.matrix_calculations import build_density_matrix, build_density_matrix_1d


class Trainer:
    """
    Train a per-bucket generative model for the β+/bg classification task. Each bucket
    has a different set of features (bucket 1: delta_E, bucket 2: delta_E & ARM,
    bucket 3: delta_E & ARM + delta_E & anni), so we build one independent model per bucket.
    Each model consists of a set of density terms (1D or 2D) and a class prior. The density
    terms are estimated from the training data using histogramming with Laplace smoothing.
    The class prior is estimated from the full per-bucket class counts.
    """

    def __init__(
        self,
        n_bins: dict[int, int] | None = None,
        joint_smoothing: float = 0.5,
        marginal_smoothing: float = 0.0,
    ) -> None:
        self.n_bins = n_bins if n_bins is not None else {1: 25, 2: 8, 3: 35}
        self.joint_smoothing = joint_smoothing  # Laplace pseudo-counts for the 2D joints (sparse-bucket safe)
        self.marginal_smoothing = marginal_smoothing  # for the 1D delta_E marginal

    def fit(self, train: dict[str, dict[int, dict[str, torch.Tensor]]]) -> "Trainer":
        # One independent model per hit-multiplicity bucket.
        self.models = {b: self.fit_bucket(train["bdecay"][b], train["bg"][b], b) for b in BUCKETS}
        return self

    def finite(self, *cols: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(cols[0], dtype=torch.bool)
        for c in cols:
            mask = mask & torch.isfinite(c)
        return mask

    def fit_bucket(self, bd: dict[str, torch.Tensor], bg: dict[str, torch.Tensor], bucket: int) -> dict:
        n_beta = int(bd["delta_E"].numel())
        n_bg = int(bg["delta_E"].numel())
        model = {"n_beta": n_beta, "n_bg": n_bg, "terms": []}
        if n_beta == 0 or n_bg == 0:
            return model  # prior-only: cannot estimate class-conditional densities

        n_bins = self.n_bins[bucket]
        if bucket == 1:
            self.add_1d(model, bd, bg, "delta_E", n_bins)
        elif bucket == 2:
            self.add_2d(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins)
        else:  # bucket 3
            self.add_2d(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins)
            self.add_2d(model, bd, bg, "delta_E", "anni", "linear", None, n_bins)
        return model

    def add_1d(
        self,
        model: dict,
        bd: dict[str, torch.Tensor],
        bg: dict[str, torch.Tensor],
        feat: str,
        n_bins: int,
    ) -> None:
        comb = torch.cat([bd[feat], bg[feat]])
        comb = comb[self.finite(comb)]
        if comb.numel() == 0:
            return  # no finite values to estimate from -> prior-only
        _, x_bins = build_density_matrix_1d(comb, n_bins_x=n_bins, spacing_x="log", log_x_floor=1e-3)
        beta, _ = build_density_matrix_1d(
            bd[feat][self.finite(bd[feat])], x_bins=x_bins, smoothing=self.marginal_smoothing
        )
        bgm, _ = build_density_matrix_1d(
            bg[feat][self.finite(bg[feat])], x_bins=x_bins, smoothing=self.marginal_smoothing
        )
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
        n_bins: int,
    ) -> None:
        cx = torch.cat([bd[xfeat], bg[xfeat]])
        cy = torch.cat([bd[yfeat], bg[yfeat]])
        m = self.finite(cx, cy)
        if int(m.sum()) == 0:
            return  # no finite (x, y) pairs to estimate from -> prior-only
        _, x_bins, y_bins = build_density_matrix(
            cx[m],
            cy[m],
            spacing_x="log",
            spacing_y=spacing_y,
            log_x_floor=1e-3,
            log_y_floor=floor_y,
            n_bins_x=n_bins,
            n_bins_y=n_bins,
        )
        mb = self.finite(bd[xfeat], bd[yfeat])
        mg = self.finite(bg[xfeat], bg[yfeat])
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
