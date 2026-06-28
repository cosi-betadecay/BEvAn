---
name: annihilation-angle-weaknesses
description: "Measured weaknesses of the annihilation_angle feature on Activation.sim (2026-06-21): it's largely a hit-multiplicity proxy (back-to-back min-cos saturates to -1 on busy events), and its 1022 keV energy prior mismatches the 511-based label / actual signal energies."
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a4769c0-c4dd-4b11-b3f8-5f862b7c1e69
---

After deleting LOR (the deleted LOR feature), audited `annihilation_angle`
(`physics/physics_factors.py:134`) against `Activation.sim`. The feature =
min over all pairwise inter-hit displacement-vector cosines (most anti-parallel
≈ -1), per subset (size>=3), with a soft Gaussian penalty pushing ΣE→1022 keV,
then min over subsets. Lower (more negative) = more annihilation-like.

**Method:** used `utils/sim_text_reader.py` (pure-Python, no MEGAlib; its
`event_is_bdecay` replicates the production label, ref 511, tol≈2.87 keV =
3σ of 2.25 FWHM). Per event computed multiplicity, label, total deposited
energy, and a PROXY for the feature: min back-to-back cosine over ALL hits
(no subset scan, no penalty). Aggregated by class AND by matched multiplicity
— the matched-n comparison is what separates "real geometry" from "just hit
count." 4478 signal / 48453 bg.

**Findings:**
1. **Multiplicity proxy / saturation (dominant).** Back-to-back min-cos at
   matched n: n3-4 sig -0.697 vs bg -0.404 (separates); n5-7 -0.990 vs -0.975;
   n8-16 -1.000 vs -0.999 (dead). Real discrimination lives ONLY in 3-4 hit
   events. Signal median 5 hits, bg median 1 → much of the feature's apparent
   power is multiplicity leaking through saturation, not back-to-back physics.
2. **1022 keV prior mismatch.** Signal ΣE_total: 2.1% near 511, 4.0% near 1022,
   **71.6% >1060** (activation bundles nuclear de-excitation gammas with the
   annihilation photons). The label is single-photon 511-based; the prior
   targets the 1022 pair — aimed at an object that barely exists here. The
   extra gammas are also the hits that manufacture the spurious back-to-back
   pairs in finding 1.
3. **NaN on 75% of bg** (needs >=3 hits). Fine as "not annihilation-like," but
   means the feature only acts on multi-hit events = the saturating ones.

**Caveat:** proxy ≠ exact feature (no ≤7 subset scan, no penalty; ΣE is
whole-event). Subsetting can only ADD spurious back-to-back chances and the
penalty is soft (λ=0.3, σ=80) applying to <4% of signal, so the proxy likely
UNDERstates the problem. Confirm on the real pipeline (Linux/MEGAlib,
[[no-megalib-on-this-mac]]) before committing a rewrite.

**Vertex-anchored back-to-back prototype (2026-06-21, parser test).** Designed
from sim truth: of 12565 annihilations only **29% deposit both photons** (hard
ceiling); when both caught, cos(V->P1,V->P2) median -1.0 (clean); vertex within
2mm of a hit 95% of the time (anchor on a hit); per-arm E partial (median 364,
only 13% both ~511) BUT **arm1+arm2 > 511 in 78%** — single Compton photon caps
at 511, so the energy-SUM>511 gate rejects single-photon collinear fakes (the
old feature's main FP). Prototype = for each hit as vertex, most anti-parallel
arm pair, assign hits to 2 arms, gate E_arms>~508, deficit-only penalty toward
1022, min over vertices. Result vs old raw min-cos: overall AUC 0.8908 vs 0.8858
(+0.005); **matched-n the real win is n3-4: 0.675 vs 0.609 (+0.066)**, but n>=5
collapses to ~0 gain (vertex ambiguity/contamination). Coverage signal 93.5% /
bg 20.5% (physical: structure present or not — keep it, don't normalize away).
Confirms back-to-back is real but WEAK per-event here = physics ceiling holds.
Prototype: /tmp/b2b_prototype.py (parser-based; NOT yet torch/pipeline). AUC !=
F1 through the density model — needs real-pipeline test on Linux to decide.

**Proposed fixes (earlier, partially superseded by the prototype above):** (1) PRIMARY — multiplicity-normalize
the back-to-back score (z-score / percentile vs a per-n null from random
isotropic hits) so busy events don't auto-win; this targets the dominant
confound. (2) Resolve energy scope: annihilation_angle's job is geometry, so
either drop the energy penalty (delta_E already owns 511) or keep ~1022 but
make it actually isolate the pair — note a true pair test needs a vertex
constraint, which is LOR territory the user chose not to keep.
