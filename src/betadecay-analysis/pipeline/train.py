import os

import torch
from dataset.datasets import BUCKETS
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
)

from pipeline.eval import Evaluator

#   bucket 1: P(delta_E)                                   [1D]
#   bucket 2: P(delta_E, ARM)                              [2D]
#   bucket 3 (default):  P(delta_E, ARM) * P(delta_E, anni)            [2D x2]
#   bucket 3 (factored): P(delta_E) * P(ARM|delta_E) * P(anni|delta_E) [1D + 2 cond]
#
# The default bucket-3 likelihood multiplies two joints that BOTH carry delta_E,
# so the delta_E likelihood ratio is counted twice — over-amplifying R toward
# beta for delta_E~511 background, which is exactly the residual n3 FP profile.
# Setting BETADECAY_FACTOR_DELTAE=1 switches to the textbook factorization
# (delta_E counted once, ARM/anni as conditionals), an opt-in A/B so the shipped
# model is unchanged by default. Assumes ARM _|_ anni | delta_E.
_FACTOR_DELTAE = os.environ.get("BETADECAY_FACTOR_DELTAE", "") not in ("", "0", "false", "False")

_JOINT_SMOOTHING = 0.5  # Laplace pseudo-counts for the 2D joints (sparse-bucket safe)
_MARGINAL_SMOOTHING = 0.0  # for the 1D delta_E marginal
_N_BINS = {1: 25, 2: 8, 3: 35}


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
        elif not _FACTOR_DELTAE:  # bucket 3, default: two joints (delta_E counted twice)
            self._add_2d(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins)
            self._add_2d(model, bd, bg, "delta_E", "anni", "linear", None, n_bins)
        else:  # bucket 3, factored: P(delta_E) * P(arm|delta_E) * P(anni|delta_E)
            self._add_bucket3_factored(model, bd, bg, n_bins)
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

    def _add_bucket3_factored(self, model, bd, bg, n_bins):
        """Bucket-3 likelihood factored so delta_E is counted exactly once:

            R = [P_beta(dE)/P_bg(dE)]
              * [P_beta(arm|dE)/P_bg(arm|dE)]
              * [P_beta(anni|dE)/P_bg(anni|dE)]

        A single shared delta_E binning is used across the marginal and both
        conditionals, so P(dE)*P(y|dE) reconstructs each joint exactly — the only
        difference from the default model is that the duplicated delta_E ratio is
        divided out. The evaluator reads conditionals through the same matrix
        lookup as joints, so the term dicts keep ``kind`` 1d / 2d unchanged.
        """
        cde = torch.cat([bd["delta_E"], bg["delta_E"]])
        cde = cde[_finite(cde)]
        if cde.numel() == 0:
            return  # prior-only
        _, x_bins = build_density_matrix_1d(
            cde, n_bins_x=n_bins, spacing_x="log", log_x_floor=1e-3
        )

        # delta_E marginal (counted once).
        beta_de, _ = build_density_matrix_1d(
            bd["delta_E"][_finite(bd["delta_E"])], x_bins=x_bins, smoothing=_MARGINAL_SMOOTHING
        )
        bg_de, _ = build_density_matrix_1d(
            bg["delta_E"][_finite(bg["delta_E"])], x_bins=x_bins, smoothing=_MARGINAL_SMOOTHING
        )
        model["terms"].append(
            {"kind": "1d", "xfeat": "delta_E", "x_bins": x_bins, "beta": beta_de, "bg": bg_de}
        )

        # Conditionals P(y | delta_E) on the shared delta_E bins.
        self._add_conditional(model, bd, bg, "delta_E", "arm", "log", 1e-3, n_bins, x_bins)
        self._add_conditional(model, bd, bg, "delta_E", "anni", "linear", None, n_bins, x_bins)

    def _add_conditional(self, model, bd, bg, xfeat, yfeat, spacing_y, floor_y, n_bins, x_bins):
        """Append a P(yfeat | xfeat) conditional term on pre-built ``x_bins``.

        The per-class joint is built on the shared x_bins (and y_bins derived from
        the combined finite pairs), then row-normalized to P(y|x) via
        ``conditional_from_joint``. Lookup is identical to a joint term.
        """
        cx = torch.cat([bd[xfeat], bg[xfeat]])
        cy = torch.cat([bd[yfeat], bg[yfeat]])
        m = _finite(cx, cy)
        if int(m.sum()) == 0:
            return
        _, _, y_bins = build_density_matrix(
            cx[m], cy[m], x_bins=x_bins, spacing_y=spacing_y, log_y_floor=floor_y, n_bins_y=n_bins
        )
        mb = _finite(bd[xfeat], bd[yfeat])
        mg = _finite(bg[xfeat], bg[yfeat])
        beta_joint, _, _ = build_density_matrix(
            bd[xfeat][mb], bd[yfeat][mb], x_bins=x_bins, y_bins=y_bins, smoothing=_JOINT_SMOOTHING
        )
        bg_joint, _, _ = build_density_matrix(
            bg[xfeat][mg], bg[yfeat][mg], x_bins=x_bins, y_bins=y_bins, smoothing=_JOINT_SMOOTHING
        )
        model["terms"].append(
            {
                "kind": "2d",
                "xfeat": xfeat,
                "yfeat": yfeat,
                "x_bins": x_bins,
                "y_bins": y_bins,
                "beta": conditional_from_joint(beta_joint, axis=0),
                "bg": conditional_from_joint(bg_joint, axis=0),
            }
        )
