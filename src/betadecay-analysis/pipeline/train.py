import torch
from dataset.datasets import BUCKETS
from modeling.matrix_calculations import build_density_matrix, build_density_matrix_1d

from pipeline.eval import Evaluator

# Per-bucket factor layout. Each bucket uses only the features physically
# available to it; delta_E is double-counted in bucket 3 (it appears in both 2D
# joints), which the campaign found load-bearing for recall.
#   bucket 1: P(delta_E)                                   [1D]
#   bucket 2: P(delta_E, ARM)                              [2D]
#   bucket 3: P(delta_E, ARM) and P(delta_E, anni)         [2D x2]
_JOINT_SMOOTHING = 0.5  # Laplace pseudo-counts for the 2D joints (sparse-bucket safe)
_MARGINAL_SMOOTHING = 1.0  # for the 1D delta_E marginal
# Per-bucket bin resolution. The low-n buckets hold only a few hundred signal
# events; a fine 25x25 grid leaves their signal density mostly smoothing (≈uniform),
# so the correct ~27:1 background prior crushed borderline 2-hit signal (recall
# 0.68). Coarse grids let those events actually shape the likelihood. n=3 is
# signal-rich and stays fine.
_N_BINS = {1: 25, 2: 10, 3: 25}


def _finite(*cols):
    mask = torch.ones_like(cols[0], dtype=torch.bool)
    for c in cols:
        mask = mask & torch.isfinite(c)
    return mask


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, train):
        # One independent model per hit-multiplicity bucket.
        self.models = {b: self._fit_bucket(train["bdecay"][b], train["bg"][b], b) for b in BUCKETS}
        Evaluator(self).evaluate(train, split_name="train")
        return self

    def _fit_bucket(self, bd, bg, bucket):
        """Build the density terms + class prior for one bucket.

        Returns ``{"n_beta", "n_bg", "terms"}`` where each term is a generic
        spec the evaluator can look up without knowing the bucket layout. Priors
        use the FULL per-bucket class counts (incl. NaN-feature events); the
        densities are estimated from finite values only.
        """
        n_beta = int(bd["delta_E"].numel())
        n_bg = int(bg["delta_E"].numel())
        model = {"n_beta": n_beta, "n_bg": n_bg, "terms": []}
        if n_beta == 0 or n_bg == 0:
            return model  # prior-only: cannot estimate class-conditional densities

        n_bins = _N_BINS[bucket]
        if bucket == 1:
            self._add_1d(model, bd, bg, "delta_E", n_bins)
        elif bucket == 2:
            self._add_2d(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins)
        else:  # bucket 3
            self._add_2d(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins)
            self._add_2d(model, bd, bg, "delta_E", "anni", "linear", None, n_bins)
        return model

    def _add_1d(self, model, bd, bg, feat, n_bins):
        comb = torch.cat([bd[feat], bg[feat]])
        comb = comb[_finite(comb)]
        if comb.numel() == 0:
            return  # no finite values to estimate from -> prior-only
        _, x_bins = build_density_matrix_1d(comb, n_bins_x=n_bins, spacing_x="log", log_x_floor=1e-3)
        beta, _ = build_density_matrix_1d(
            bd[feat][_finite(bd[feat])], x_bins=x_bins, smoothing=_MARGINAL_SMOOTHING
        )
        bgm, _ = build_density_matrix_1d(
            bg[feat][_finite(bg[feat])], x_bins=x_bins, smoothing=_MARGINAL_SMOOTHING
        )
        model["terms"].append(
            {"kind": "1d", "xfeat": feat, "x_bins": x_bins, "beta": beta, "bg": bgm}
        )

    def _add_2d(self, model, bd, bg, xfeat, yfeat, spacing_y, floor_y, n_bins):
        cx = torch.cat([bd[xfeat], bg[xfeat]])
        cy = torch.cat([bd[yfeat], bg[yfeat]])
        m = _finite(cx, cy)
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
        mb = _finite(bd[xfeat], bd[yfeat])
        mg = _finite(bg[xfeat], bg[yfeat])
        beta, _, _ = build_density_matrix(
            bd[xfeat][mb], bd[yfeat][mb], x_bins=x_bins, y_bins=y_bins, smoothing=_JOINT_SMOOTHING
        )
        bgm, _, _ = build_density_matrix(
            bg[xfeat][mg], bg[yfeat][mg], x_bins=x_bins, y_bins=y_bins, smoothing=_JOINT_SMOOTHING
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
