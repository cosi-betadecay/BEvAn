"""Ablation A2 — energy only (strip the model to the single delta_E term).

What it is:
    The full Bayesian per-bucket machinery kept intact, but every model reduced
    to the lone delta_E density term: no ARM, no annihilation-angle, no geometry
    of any kind. Classification still flows through the same likelihood-ratio /
    prior / calibration path — only the feature set collapses to "distance of the
    summed subset energy from 511 keV".

What it tries to prove:
    How much of the champion's performance is just the 511 keV energy window. The
    expected result is the near-tie already seen on the VM (delta_E-only ~0.8910
    vs full 0.8935), i.e. the entire Compton-geometry apparatus buys ~0.0025 F1.
    Establishing this honestly is the spine of the paper, not a footnote.

Why it is important:
    The ground truth is itself a photopeak criterion (an ANNI secondary depositing
    within 3 sigma of 511 keV), so a reviewer's first attack is circularity: "your
    label and your headline feature are the same quantity — what did the Compton
    machinery add?" This row answers it with a number instead of a hand-wave, and
    reframes the contribution from "we built a strong classifier" to "we quantify
    how little geometry adds at the photopeak, and explain why."
"""
