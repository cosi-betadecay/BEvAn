"""Ablation A4 — no annihilation angle (drop the back-to-back term).

What it is:
    The champion model with the 2D (delta_E, anni) joint removed from bucket 3, so
    bucket 3 collapses to the same (delta_E, ARM) form as bucket 2. The
    vertex-anchored back-to-back score — the feature most directly motivated by
    the physics of two anti-parallel 511 keV photons — is taken out of the model.

What it tries to prove:
    The marginal value of the annihilation-angle hypothesis. anni is the most
    "physically on-the-nose" feature in the model (it explicitly tests the
    back-to-back signature of positron annihilation), so quantifying its
    contribution is the cleanest test of whether the defining signature of the
    signal is actually separable in COSI data or whether it is washed out by
    partial energy deposition and Compton-telescope geometry.

Why it is important:
    For a method/instrument paper, every term must earn its place. If the feature
    that most literally encodes the signal physics adds little (as the n3 results
    suggest), that is a substantive finding about the detectability limit, not a
    reason to hide the term. This row supplies the evidence either way.
"""
