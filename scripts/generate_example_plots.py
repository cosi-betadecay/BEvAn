#!/usr/bin/env python3
"""Regenerate the styling-reference plot drafts in ``docs/plots/``.

These figures use *synthetic* data and exist only to document the look of
``utils/plots.py`` (palette, layouts, labels). The real figures come from the
eval pipeline and are logged to W&B via ``utils/wandb_logging.py``. No
MEGAlib/ROOT is needed: the plot helpers take plain arrays and term dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

# Make the src package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "betadecay-analysis"))

from utils.plots import (
    plot_confusion_matrix,
    plot_density_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "plots"


def main() -> None:
    """Render the five reference figures into ``docs/plots/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # ROC / PR / confusion: imbalanced task (~12% signal), signal scores higher.
    n = 4000
    labels = rng.random(n) < 0.12
    scores = rng.normal(0, 1, n) + labels * 1.7
    plot_roc_curve(scores, labels).savefig(OUT_DIR / "example_roc.png", dpi=110, bbox_inches="tight")
    plot_pr_curve(scores, labels).savefig(OUT_DIR / "example_pr.png", dpi=110, bbox_inches="tight")
    plot_confusion_matrix(583, 118, 21, 5874).savefig(
        OUT_DIR / "example_confusion_matrix.png", dpi=110, bbox_inches="tight"
    )

    # 1D density term: delta_E (bucket 1), signal near the 511 keV photopeak.
    xb1 = torch.logspace(-3, 3, 26)
    beta1 = torch.tensor(
        np.histogram(np.abs(rng.normal(0, 0.05, 6000)).clip(1e-3, 1e3), bins=xb1.numpy())[0],
        dtype=torch.float32,
    )
    bg1 = torch.tensor(np.histogram(10 ** rng.uniform(-3, 3, 8000), bins=xb1.numpy())[0], dtype=torch.float32)
    plot_density_matrix(
        {"kind": "1d", "xfeat": "delta_E", "x_bins": xb1, "beta": beta1 / beta1.sum(), "bg": bg1 / bg1.sum()}
    ).savefig(OUT_DIR / "example_density_1d.png", dpi=110, bbox_inches="tight")

    # 2D density term: (delta_E, ARM) (bucket 2).
    xb, yb = torch.logspace(-1, 3, 21), torch.logspace(-3, 0, 21)
    sx = np.abs(rng.normal(0, 0.4, 5000)).clip(0.1, 1e3)
    sy = (10 ** rng.uniform(-3, 0.2, 5000)).clip(1e-3, 1)
    bx = (10 ** rng.uniform(-1, 3, 9000)).clip(0.1, 1e3)
    by = (10 ** rng.uniform(-3, 0, 9000)).clip(1e-3, 1)
    beta2 = torch.tensor(np.histogram2d(sx, sy, bins=[xb.numpy(), yb.numpy()])[0], dtype=torch.float32)
    bg2 = torch.tensor(np.histogram2d(bx, by, bins=[xb.numpy(), yb.numpy()])[0], dtype=torch.float32)
    plot_density_matrix(
        {
            "kind": "2d",
            "xfeat": "delta_E",
            "yfeat": "arm",
            "x_bins": xb,
            "y_bins": yb,
            "beta": beta2 / beta2.sum(),
            "bg": bg2 / bg2.sum(),
        }
    ).savefig(OUT_DIR / "example_density_2d.png", dpi=110, bbox_inches="tight")

    print(f"Wrote 5 reference plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
