---
name: energy-specificity-vs-generic-geometry
description: "WHY energy dominates the classifier — 511 keV is the only β+-specific signature (a spectral line); ARM/anni are generic Compton properties shared by the activation-decay background, so geometry is weak+redundant by design, not broken"
metadata: 
  node_type: memory
  type: project
  originSessionId: dfb1c7a1-485d-40c8-8fbd-d07d598dc9f4
---

The branch-75 ablation study (2026-06-24) settled WHY energy dominates and the
Compton geometry does not — a clean physical reading (user's; I initially
mis-framed it and was corrected):

**511 keV is the ONLY β+-specific observable** — a spectral *line* unique to
positron annihilation. The COSI background is the activation cocktail
(cosmic-proton-activated isotopes, `data/cosima/Activation*`) = OTHER real decays.
Their photons Compton-scatter with valid kinematics (good ARM) and multi-scatter
can sit back-to-back, so ARM and the annihilation-angle are **generic** properties
of any well-reconstructed photon event, shared by signal and background alike.
Geometry therefore cannot separate β+ from the decay background. And on true signal,
energy already entails the geometry (same annihilation) → geometry is **redundant**
on top of energy.

Evidence — redesigned `factor_contributions` (≥3-hit / bucket-3 common population,
clean 1D density per factor, fit on train / scored on eval): ΔE-alone ≈ full model
(GeACT 0.862 vs 0.863, Max 0.901 vs 0.902, NCT 0.875 vs 0.875 F1); ARM/anni
standalone AUC ~**0.60** (barely above chance). The ~0.60 is the EXPECTED baseline
*because the background is other decays* — NOT a surprising "specificity gap" needing
a separate explanation (that was my framing error; geometry was never going to be
strong standalone).

Suite context: across `no_calibration` / `learned_weights` / `no_ckd_order`,
**ΔAUC ≤ 0.001** — nothing moves the ranking; the model sits at a stable AUC ceiling
and design choices only shuffle the operating point. `learned_weights` (threshold now
calibrated on train logits): ΔF1 ≈ −0.009, ΔAUC ≈ +0.0006 → equal weights optimal,
learned weights overfit ([[exhausted-levers]]). `no_ckd_order`: CKD ≈ energy
ordering (ΔF1 +0.0013 favoring energy) → CKD doesn't earn its keep. `no_calibration`:
ΔAUC = 0 exactly (threshold-only lever; +0.002 F1 on Ge, big on balloon).

**Why:** energy is powerful because it is *process-specific* (a line), not because
geometry is broken; geometry is weak because it is *generic* to all photon physics in
a decay-rich background. This is the paper's central physical claim.

**How to apply:** Frame the paper as "the working discriminator is the process-specific
511 keV annihilation line; the Compton-geometry features describe scattering in general,
which the activation background shares, so they neither separate β+ alone nor add to the
energy that already implies them." Do NOT claim geometry "helps a little" from the
≥3-hit figure (ΔE-alone matches full there). Do NOT delete ARM/anni — their joints
double-count ΔE which is load-bearing ([[exhausted-levers]]: n3 0.8935→0.8681
if dropped); their weakness is a finding to REPORT, and the honest simplification is to
model ΔE explicitly rather than drop geometry. Refines [[exhausted-levers]] and
[[annihilation-angle-weaknesses]]; the ablation harness lives in `ablations/`.
