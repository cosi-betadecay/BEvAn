---
name: naive-bayes-dependence-accepted
description: "User accepts that features violate naive-Bayes independence in practice; don't reject a feature for overlapping with ARM/ΔE"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5aa736b2-2f4d-42fc-933b-64f2a37e6253
---

The classifier is naive Bayes, so in theory the feature terms should be conditionally independent given the class. The user explicitly knows this is impossible to satisfy in practice and is **fine with feature dependence** — e.g. an annihilation/back-to-back feature whose discriminating power leans on Compton physics that overlaps ARM, or ARM/anni terms that partly double-count ΔE.

**Why:** The features are physics-motivated likelihood terms (511 keV / Compton kinematics), and the real signal is concentrated in energy (511 line), so geometric terms necessarily share information with each other and with ΔE. Demanding true independence would forbid most useful features. The user cares about whether a feature *helps in practice* (and is physically defensible), not whether it respects the naive-Bayes independence assumption.

**How to apply:** Do NOT veto or hedge a feature design merely because it correlates with / double-counts an existing term ([[exhausted-levers]], [[energy-specificity-vs-generic-geometry]]). Dependence is expected and accepted — note it if load-bearing (e.g. the n3 ΔE double-count in [[hit-multiplicity-bucketing]] is load-bearing), but judge features on measured F1/AUC impact and physics defensibility ([[cosi-working-style]]), not on independence purity.
