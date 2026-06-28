---
name: exhausted-levers
description: "Don't-retry guardrail: per-event levers that were measured and exhausted (weighting, energy noising, ARM-as-double-count, postprocessing, balloon base-rate, PET/LOR vertex). Each cost ~0 F1 or was refuted. Consult before re-proposing any of these."
metadata:
  node_type: memory
  type: project
  originSessionId: consolidation-2026-06-28
---

Levers that were measured to exhaustion (mostly on the VM, 2026-06-22) and should
NOT be re-tried without a genuinely new angle. They are recorded so the same dead
ends don't get re-walked. The per-event model sits at its ceiling
([[betadecay-classifier-state]]); these are why.

- **Term weighting / logistic LLR weights** — equal weights (1,1,1) ARE the MLE
  optimum (logistic-regression proof, not assertion): best-F1 swept 0.8947 vs
  logistic 0.8925 = overfit. Threshold/weighting/R all floored. Don't re-propose
  learned weights.
- **Un-noised energy degeneracy** — REFUTED: production `MFileEventsSim` NOISES
  energies (signal ΔE median 0.46, not 0.001), so ΔE-only=0.8910 < full 0.8935.
  KEEP ARM/anni. The parser's clean energies ≠ production — never trust a parser-only
  energy result without VM re-measure.
- **ARM = ΔE double-count (old far-field ARM)** — dropping the (ΔE,arm) joint cost
  0.8935→0.8681, but the loss was all the ΔE double-count, not geometry (n2 ARM
  contributed ~0). NOTE: superseded — branch-75 reworked ARM into the signed
  511-aware feature that is now the champion ([[betadecay-classifier-state]]).
- **Postprocessing / within-event levers** — .tra cross-ref DEAD (structural);
  CKD/energy-ordering ≈ energy. The only noising-independent win was ε·β population
  correction (the science FoM, validated <0.5% bias) — that's analysis, not a feature.
- **Balloon AUC≈0.99 / F1≈0.15** — base-rate arithmetic (0.14% prevalence), not
  broken densities: best_f1 ≈ (1−AUC)·n_bg/n_sig predicts it exactly. Calibration
  3×'d the F1 purely by threshold; the model was identical. Don't "fix the densities."
- **PET/LOR vertex annihilation rebuild** — vertex IS solvable via near-field/PET
  (validated vs .sim truth), but DEAD as a feature: residual AUC 0.48 (below chance)
  once partitioned, because back-to-back geometry carries no β⁺-specific signal that
  ΔE doesn't already have (ΔE alone = 0.997 on those events). Not wired in.

Common thread: 511 keV (ΔE) is the only β⁺-*specific* signal; geometry/back-to-back
features are generic Compton properties shared by the activation background, so they
are weak and redundant. See [[energy-specificity-vs-generic-geometry]].
