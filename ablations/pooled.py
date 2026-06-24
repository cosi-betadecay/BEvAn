"""Ablation A5 — pooled model (collapse the three hit-multiplicity buckets into one).

What it is:
    The champion routes each event to one of three models by which features it can
    produce (bucket 1: delta_E only; bucket 2: + ARM; bucket 3: + anni), each with
    its own densities and class prior. This row removes that routing: a single
    pooled model with one prior over all events, so the per-bucket structure is
    gone while the feature definitions stay the same.

What it tries to prove:
    Whether hit-multiplicity bucketing earns its complexity. The honest expectation
    is mixed — overall F1 is pooled-dominated and dragged by bucket 2 rather than
    bucket 3 — so this row shows how much the per-bucket priors and per-bucket
    density resolution actually buy over a single flat model.

Why it is important:
    "Why three buckets?" is the obvious reviewer question about the model's
    structure. A method paper has to justify its moving parts; this ablation either
    defends the bucketing with a number or honestly bounds how marginal it is,
    which is itself a useful statement about where the information lives.
"""
