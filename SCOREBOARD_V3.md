# Scoreboard V3 — Two-photon Joint Angle/Energy Constraint

Loop surface: `annihilation_angle` in `physics/physics_factors.py` (body only,
iter ≥1). Lever: restrict which candidate pairs/triples are eligible for the
cosine min using the per-hit energies threaded in by Phase 0, so the chosen
back-to-back pair must also be energy-consistent with a ~1022 keV annihilation
total (theory §7).

Best so far: iter 2, MCC=0.8792 (soft Gaussian energy penalty, λ=0.3, σ=80)

**Baseline recall = 0.9536. Guardrail floor = 0.9036** (recall ≥ baseline − 0.05).
Stop if recall < 0.9036 for two consecutive iters.

MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)), computed from the
eval confusion matrix printed by `predict`.

| Run ID | Timestamp (UTC) | Status | Hypothesis | What changed | MCC | Δ vs best | F1 | Precision | Recall | FPR | TP / FP / FN / TN | Notes / next idea |
|--------|-----------------|--------|------------|--------------|-----|-----------|----|-----------|--------|-----|--------------------|-------------------|
| 0 | 2026-06-09T21:30:00Z | baseline | — | post-Phase-0 plumbing: `energies` threaded into `annihilation_angle` as a no-op pass-through | 0.8785 | — | 0.8880 | 0.8309 | 0.9536 | 0.0195 | 575 / 117 / 28 / 5875 | **V3 baseline ≡ V2 iter 13, byte-identical → Phase 0 confirmed behavior-preserving.** Energies now reach `annihilation_angle` per subset (shape (B,N) matching positions). From iter 1 only the `annihilation_angle` body (+ module-level helpers in the same file) is editable. The residual 117 FP are background that is 511-like in all three features independently; V3's thesis is that coupling angle and energy at the *same* hit-pair (1022 keV joint gate) is asymmetric on signal vs random Compton coincidences. First hypothesis: hard 1022 keV pair gate (seed #1). |
| 1 | 2026-06-09T21:45:00Z | reverted | hard 1022 keV subset gate (seed #1): only subsets with ΣE ∈ 1022±80 keV eligible for the back-to-back cosine min; drop the size group (NaN) if none qualify | `annihilation_angle` sized branch: per-subset `keep = \|ΣE−1022\|≤80`, filter `group_positions` before the cosine min; new module consts `ANNI_TOTAL_KEV=1022`, `ANNI_SIGMA_E=80` | 0.8615 | -0.0170 | 0.8741 | 0.8463 | 0.9038 | 0.0165 | 545 / 99 / 58 / 5893 | **WORSE: TP −30, FP −18 (~5:3, net negative).** The gate IS asymmetric (FP fell, precision 0.831→0.846, FPR 0.0195→0.0165) — energy coupling carries real discriminating signal — but a HARD subset cut is too destructive: recall collapsed 0.9536→0.9038 (at the 0.9036 floor). Mechanism: for ~30 signal events the most-back-to-back subset does NOT sum to ~1022 (second annihilation photon escaped / partially deposited / captured in a different-size subset), and even for kept events, removing off-energy subsets *raises* the per-size min cosine (less back-to-back) → genuine signal crosses the boundary. **Lesson (mirrors V2's "filter, don't remove"): hard-dropping subsets is recall-dangerous because it raises signal's own angle min.** Reverted. Next (iter2): soft Gaussian energy penalty (seed #2) at subset granularity — keep EVERY subset eligible (no recall cliff) but add λ·(1−exp(−(ΣE−1022)²/(2σ²))) to each subset's contributed cosine so energy-inconsistent back-to-back configs are nudged UP while on-energy ones keep their low cosine. Small λ to preserve the cosine scale. |
| 2 | 2026-06-09T21:47:00Z | **new best** | soft Gaussian energy penalty (seed #2), λ=0.3, σ=80: keep EVERY subset eligible, add `λ·(1−exp(−(ΣE−1022)²/(2σ²)))` to each subset's back-to-back cosine before the per-size min, clamp ≤1 | new `_per_subset_min_cosines` helper (per-subset `amin`, behavior-preserving at λ=0); sized branch adds the penalty per subset then `amin`; new const `ANNI_LAMBDA=0.3` | 0.8792 | +0.0007 | 0.8887 | 0.8321 | 0.9536 | 0.0194 | 575 / 116 / 28 / 5876 | **NEW BEST (marginal): FP −1, recall & TP unchanged.** Confirms the iter-1 lesson's fix: the soft penalty removes the recall cliff entirely (recall 0.9536 preserved exactly — no signal event crosses the boundary because nothing is dropped, only ranked) while still shaving 1 FP. Energy coupling is asymmetric and real, but at λ=0.3 the nudge is too gentle to move many bg events across an angle bin (~0.08 wide); only one off-energy bg subset stopped winning its event's min. Mechanism validated, magnitude says push harder. Next (iter3): λ sweep upward (try λ=0.8) — same σ=80, keep recall floor in view; if recall stays pinned at 0.9536 there is headroom to penalise more aggressively before any signal event's best subset is off-energy enough to matter. |

---

## Notes

- The current `annihilation_angle` minimizes `|cos + 1|` over **pairs of
  inter-hit difference vectors** (`all_vector_cosines_matrix_calculation`):
  for an N-hit subset it forms the M = C(N,2) inter-hit vectors and takes the
  most back-to-back pair of *vectors*. It is NOT pairing hits directly; the
  "pair" is two relative-direction vectors. The energy gate must therefore be
  expressed in terms compatible with that vector construction (see iter notes).
- Return convention is fixed: a cosine in [-1, 1], lower = more back-to-back =
  more signal-like. NaN ⇒ event drops out (downstream maps NaN→0 density).
- One verb per iter. Push only on a new best.
