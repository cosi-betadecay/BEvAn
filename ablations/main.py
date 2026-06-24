"""Ablation driver — run every ablation against every dataset in one pass.

What it is:
    The single entry point for the whole ablation study. It discovers every
    matched {name}.sim / {name}.tra pair in data/ (looking each dataset's geometry
    up by name, the same static GEOMETRIES convention batch_analysis.py uses, since
    each simulation was produced against a different geometry), and runs the full
    ablation set over the cross product of (ablation x dataset). Every cell is
    evaluated on the same fixed split/seed so the rows stay comparable, and each
    writes a uniform metrics record under results/ablations/<timestamp>/<ablation>/
    <dataset>/, which it then assembles into one wide summary table (the paper
    table: ablations down the rows, datasets across the columns).

    The expensive step is feature extraction (streaming events through MEGAlib/
    ROOT), so the driver extracts each dataset's features once, caches the tensors
    to disk, and lets the model-level ablations (base, deltaE_only, arm_as_deltaE,
    no_anni, pooled, no_calibration, learned_weights) reuse that cache. The one
    feature-level ablation (no_ckd_order) declares it needs re-extraction and is
    re-streamed rather than served from cache.

What it tries to prove:
    Nothing on its own — it is orchestration, not an experiment. Its job is to make
    the study reproducible and turnkey: one command regenerates every number in the
    ablation section, on every geometry, from the same data and the same split, so
    a reviewer (or a future run) can rebuild the entire table without hand-running
    eight scripts across five datasets.

Why it is important:
    A paper's ablation table is only credible if it is internally consistent — same
    split, same eval, same baseline across every cell — and only maintainable if it
    can be regenerated wholesale when the model, the data, or a feature definition
    changes. Centralizing the cross product here (rather than per-ablation
    invocations) is what guarantees that consistency and prevents the baseline drift
    that would quietly invalidate the comparison.

Intended usage (once implemented):
    python ablations/main.py                  # all ablations x all datasets
    python ablations/main.py --ablations base deltaE_only
    python ablations/main.py --datasets SPILike Max
    python ablations/main.py --wandb           # optional logging, as elsewhere
"""
