# Scoreboard V3 — Two-photon Joint Angle/Energy Constraint

Loop surface: `annihilation_angle` in `physics/physics_factors.py` (body only,
iter ≥1). Lever: restrict which candidate pairs/triples are eligible for the
cosine min using the per-hit energies threaded in by Phase 0, so the chosen
back-to-back pair must also be energy-consistent with a ~1022 keV annihilation
total (theory §7).

**HALTED 2026-06-10 — best: iter 2, MCC=0.8792 (soft Gaussian energy penalty,
λ=0.3, σ=80). See halt summary below the table.**

**Baseline recall = 0.9536. Guardrail floor = 0.9036** (recall ≥ baseline − 0.05).
Stop if recall < 0.9036 for two consecutive iters.

MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)), computed from the
eval confusion matrix printed by `predict`.

| Run ID | Timestamp (UTC) | Status | Hypothesis | What changed | MCC | Δ vs best | F1 | Precision | Recall | FPR | TP / FP / FN / TN | Notes / next idea |
|--------|-----------------|--------|------------|--------------|-----|-----------|----|-----------|--------|-----|--------------------|-------------------|
| 0 | 2026-06-09T21:30:00Z | baseline | — | post-Phase-0 plumbing: `energies` threaded into `annihilation_angle` as a no-op pass-through | 0.8785 | — | 0.8880 | 0.8309 | 0.9536 | 0.0195 | 575 / 117 / 28 / 5875 | **V3 baseline ≡ V2 iter 13, byte-identical → Phase 0 confirmed behavior-preserving.** Energies now reach `annihilation_angle` per subset (shape (B,N) matching positions). From iter 1 only the `annihilation_angle` body (+ module-level helpers in the same file) is editable. The residual 117 FP are background that is 511-like in all three features independently; V3's thesis is that coupling angle and energy at the *same* hit-pair (1022 keV joint gate) is asymmetric on signal vs random Compton coincidences. First hypothesis: hard 1022 keV pair gate (seed #1). |
| 1 | 2026-06-09T21:45:00Z | reverted | hard 1022 keV subset gate (seed #1): only subsets with ΣE ∈ 1022±80 keV eligible for the back-to-back cosine min; drop the size group (NaN) if none qualify | `annihilation_angle` sized branch: per-subset `keep = \|ΣE−1022\|≤80`, filter `group_positions` before the cosine min; new module consts `ANNI_TOTAL_KEV=1022`, `ANNI_SIGMA_E=80` | 0.8615 | -0.0170 | 0.8741 | 0.8463 | 0.9038 | 0.0165 | 545 / 99 / 58 / 5893 | **WORSE: TP −30, FP −18 (~5:3, net negative).** The gate IS asymmetric (FP fell, precision 0.831→0.846, FPR 0.0195→0.0165) — energy coupling carries real discriminating signal — but a HARD subset cut is too destructive: recall collapsed 0.9536→0.9038 (at the 0.9036 floor). Mechanism: for ~30 signal events the most-back-to-back subset does NOT sum to ~1022 (second annihilation photon escaped / partially deposited / captured in a different-size subset), and even for kept events, removing off-energy subsets *raises* the per-size min cosine (less back-to-back) → genuine signal crosses the boundary. **Lesson (mirrors V2's "filter, don't remove"): hard-dropping subsets is recall-dangerous because it raises signal's own angle min.** Reverted. Next (iter2): soft Gaussian energy penalty (seed #2) at subset granularity — keep EVERY subset eligible (no recall cliff) but add λ·(1−exp(−(ΣE−1022)²/(2σ²))) to each subset's contributed cosine so energy-inconsistent back-to-back configs are nudged UP while on-energy ones keep their low cosine. Small λ to preserve the cosine scale. |
| 2 | 2026-06-09T21:47:00Z | **new best** | soft Gaussian energy penalty (seed #2), λ=0.3, σ=80: keep EVERY subset eligible, add `λ·(1−exp(−(ΣE−1022)²/(2σ²)))` to each subset's back-to-back cosine before the per-size min, clamp ≤1 | new `_per_subset_min_cosines` helper (per-subset `amin`, behavior-preserving at λ=0); sized branch adds the penalty per subset then `amin`; new const `ANNI_LAMBDA=0.3` | 0.8792 | +0.0007 | 0.8887 | 0.8321 | 0.9536 | 0.0194 | 575 / 116 / 28 / 5876 | **NEW BEST (marginal): FP −1, recall & TP unchanged.** Confirms the iter-1 lesson's fix: the soft penalty removes the recall cliff entirely (recall 0.9536 preserved exactly — no signal event crosses the boundary because nothing is dropped, only ranked) while still shaving 1 FP. Energy coupling is asymmetric and real, but at λ=0.3 the nudge is too gentle to move many bg events across an angle bin (~0.08 wide); only one off-energy bg subset stopped winning its event's min. Mechanism validated, magnitude says push harder. Next (iter3): λ sweep upward (try λ=0.8) — same σ=80, keep recall floor in view; if recall stays pinned at 0.9536 there is headroom to penalise more aggressively before any signal event's best subset is off-energy enough to matter. |
| 3 | 2026-06-09T23:43:00Z | reverted | λ sweep up (dose-response): soft Gaussian penalty strength λ 0.3→0.8, σ=80 unchanged | `ANNI_LAMBDA = 0.8` (single constant) | 0.8756 | -0.0036 | 0.8858 | 0.8333 | 0.9453 | 0.0190 | 570 / 114 / 33 / 5878 | **WORSE: TP −5, FP −2 vs iter 2.** Dose-response now bracketed: λ=0.3 → FP−1/TP−0 (+0.0007), λ=0.8 → FP−2/TP−5 (−0.0036), hard gate (λ≈∞) → FP−18/TP−30 (−0.0170). Above λ≈0.3 the marginal signal damage grows ~2.5× faster than the bg benefit. Mechanism: the penalty SATURATES at λ for any far-off-energy subset, and signal-with-escaped-photon sits exactly there — so cranking λ moves genuine signal across angle bins while far-off bg (already saturated at λ=0.3's 3-4 bin shift) gains little extra separation. The subset-total ΣE≈1022 lever is real but nearly exhausted at gentle strength. Reverted to λ=0.3. Next (iter4): λ=0.5 midpoint to pin the optimum's left edge — if it is also worse, the λ axis is closed and iter 5 moves to a structurally different seed (#3 per-photon 511 prior or #4 chain-level constraint). |
| 4 | 2026-06-09T23:46:00Z | reverted | λ=0.5 midpoint (close the dose-response bracket) | `ANNI_LAMBDA = 0.5` (single constant) | 0.8759 | -0.0033 | 0.8860 | 0.8324 | 0.9469 | 0.0192 | 571 / 115 / 32 / 5877 | **WORSE: TP −4, FP −1 vs iter 2. λ axis CLOSED.** Full curve: λ=0.3 → +0.0007, λ=0.5 → −0.0033, λ=0.8 → −0.0036, hard gate → −0.0170. Signal damage turns on sharply just above 0.3 (4 TP lost by 0.5) while bg barely moves (1 FP) — the subset-total ΣE≈1022 residual has no more asymmetry to extract at any strength. Reverted to λ=0.3. The total is the WRONG residual to push harder on: an 800+222 keV random coincidence passes ΣE≈1022 perfectly. Next (iter5): seed #3 per-photon 511 prior as a soft penalty — replace the total residual with the best two-partition SPLIT residual r² = min over hit-bipartitions (ΣA−511)² + (ΣB−511)², same λ=0.3 soft-Gaussian form. Signal subsets can split ~511/511; coincidence subsets that merely sum to 1022 cannot. |
| 5 | 2026-06-09T23:49:00Z | reverted | seed #3 per-photon 511 prior: replace total-1022 residual with best two-partition split residual r² = min over hit-bipartitions (ΣA−511)²+(ΣB−511)², soft Gaussian λ=0.3, σ=80 | new `_best_split_residual_sq` helper (2^(N−1) bipartition enumeration, hit-0 pinned; N>13 falls back to T²/2); sized branch penalty uses split_r2; new consts `ANNI_PHOTON_KEV`, `ANNI_MAX_SPLIT_HITS` | 0.8752 | -0.0040 | 0.8853 | 0.8311 | 0.9469 | 0.0194 | 571 / 116 / 32 / 5876 | **WORSE: TP −4, FP ±0 vs iter 2 — the split constraint bought ZERO bg rejection.** The 800+222-style coincidence the split residual was built to kill is apparently NOT what the residual 116 FP are made of: their winning subsets split ~511/511 fine. Combined with iter 1 (hard 1022 gate cut FP only 18/117), the picture is that much of the residual background is genuinely annihilation-like in ENERGY (activation bg produces real β⁺/511 pairs) — energy-sum constraints (total, split, any σ/λ) are near-exhausted as a discriminator. Signal cost (−4 TP) came from the split residual's implicit asymmetry punishment on partially-deposited chains. Reverted to iter-2 total residual. Next (iter6): seed #5 asymmetry exploit, ADDITIVE on top of iter-2 best: extra soft penalty on subsets whose MAX single-hit energy exceeds 511 keV (kinematically impossible for any hit belonging to either 511 photon's chain; Compton edge for 511 is 341 keV, full absorption 511) — targets in-detector pair-production / β-decay topologies that energy SUMS cannot see. λ₂=0.3, σ₂=30 on overflow=relu(maxE−511). |
| 6 | 2026-06-09T23:52:00Z | reverted | seed #5 asymmetry exploit, additive on iter-2 best: extra soft penalty λ₂=0.3, σ₂=30 on overflow=relu(max-hit−511) per subset | sized branch adds maxhit penalty term on top of total-1022 penalty; consts `ANNI_MAXHIT_LAMBDA`, `ANNI_MAXHIT_SIGMA` | 0.8752 | -0.0040 | 0.8853 | 0.8311 | 0.9469 | 0.0194 | 571 / 116 / 32 / 5876 | **WORSE: TP −4, FP ±0 — confusion matrix IDENTICAL to iter 5.** Residual bg has no >511 keV hits to catch; the same ~4 fragile signal events (winning subset carries a legitimate non-photon hit, e.g. β-ionization deposit, sitting at an angle-bin boundary) get pushed over by ANY additional penalty mass. Pattern across iters 3-6: energy-side penalties beyond iter 2's gentle total-residual nudge cost 4-5 TP and buy 0-2 FP, regardless of the residual's form (total / split / max-hit). The residual ~116 FP are energy-indistinguishable from signal at subset granularity. Reverted. Next (iter7): the one untried axis on the standing best — σ sweep on the total residual (σ 80→50, λ=0.3 fixed): sharpens the penalty gradient near 1022 so mildly-off-energy bg pays more while exactly-on-energy signal still pays ~0; signal had zero TP loss at λ=0.3/σ=80, so there may be σ headroom before the fragile-TP failure mode reappears. |
| 7 | 2026-06-09T23:55:00Z | reverted | σ sweep down on iter-2 best: total-residual σ 80→50, λ=0.3 fixed | `ANNI_SIGMA_E = 50.0` (single constant; diff vs best verified to be σ only) | 0.8782 | -0.0010 | 0.8879 | 0.8319 | 0.9519 | 0.0194 | 574 / 116 / 29 / 5876 | **WORSE (mildly): TP −1, FP ±0.** Sharpening the gradient near 1022 moved no bg (their off-energy subsets are already paying ~max penalty or have on-energy twins) and clipped 1 mildly-sub-1022 signal event. σ-down closed. Key insight for next iter: every TP loss in iters 3-7 is a signal event whose winning subset sits BELOW 1022 (partial deposit / escaped photon) — physically, signal subsets can only be sub-1022, never super-1022 (escape removes energy; nothing adds it to a pure two-photon subset, and any subset polluted by an extra hit has a penalty-free pure twin in the enumeration carrying the same back-to-back vectors). Bg coincidences land on BOTH sides. Next (iter8): one-sided penalty — residual = relu(ΣE−1022), λ=0.3, σ=80: structurally zero signal cost by construction; if TP holds at 575, the λ axis REOPENS on the safe side (iter 9 = crank λ). |
| 8 | 2026-06-09T23:58:00Z | reverted | one-sided energy-excess penalty: residual = relu(ΣE−1022), λ=0.3, σ=80 (sub-1022 subsets ride free) | sized branch: `excess = (total_E - 1022).clamp(min=0)` replaces the symmetric residual | 0.8785 | -0.0007 | 0.8880 | 0.8309 | 0.9536 | 0.0195 | 575 / 117 / 28 / 5875 | **Confusion matrix EXACTLY = iter-0 baseline.** Both halves of the hypothesis resolved cleanly: (a) signal protection CONFIRMED — TP back to 575, the iters 3-7 TP losses were all sub-1022 partial-deposit signal; (b) bg yield ZERO — no bg event's winning subset sits above 1022 (FP 117 = baseline; iter 2's −1 FP was a SUB-1022 bg subset). The planned λ-crank on the safe side is VOIDED — λ·0 = 0. Verdict on the energy axis: bg's exploitable off-energy mass lives only sub-1022, exactly where signal partial deposits live; subset-level energy penalties are a closed book (two-sided gentle = +0.0007 is all there is). Reverted. Next (iter9): the one untried granularity — PAIR-level coupling (the mission's literal statement): penalize each candidate vector pair by its own energy interpretation (disjoint pair: each vector's 2 endpoint hits sum ≈511 per chain; shared-hit pair: best of the 2 shared-hit assignments), applied INSIDE the per-subset min, so a bg subset can no longer win the angle with one hit combination while owing its energy consistency to a different partition. |
| 9 | 2026-06-10T00:01:00Z | reverted | seed #4-style PAIR-level coupling: each candidate vector pair penalized by its own 511/511 chain interpretation (disjoint pair: endpoint-hit sums per vector; shared-hit pair: best of 2 shared-hit assignments), soft Gaussian λ=0.3 σ=80 applied INSIDE the per-subset min | `_per_subset_min_cosines` extended with optional `energies`: per-pair r² from triu hit-index maps, penalty added to each pair cosine before `amin`; sized branch delegates entirely to the helper | 0.8745 | -0.0047 | 0.8846 | 0.8299 | 0.9469 | 0.0195 | 571 / 117 / 32 / 5875 | **WORSE: TP −4, FP +1 vs iter 2 (FP back at baseline 117).** The mission's most literal form — energy consistency of the chosen pair itself — is ALSO null on bg: residual FP winning pairs interpret as 511/511 (or their next-best pair wins in the same bin). TP cost from signal chains longer than 2 hits whose endpoint sums under-shoot 511. Verdict: see halt summary below. |

---

## V3 halt summary (2026-06-10, after iter 9)

**Best: iter 2, MCC = 0.8792** (soft Gaussian subset-total penalty, λ=0.3,
σ=80) — committed at `342e417`. Δ vs V2 baseline: +0.0007 (FP 117→116,
recall untouched at 0.9536).

Halted after 7 consecutive non-best iters (protocol allows 10) because the
remaining action space is empirically dominated, not merely unlucky:

1. **Upper bound established (iter 1):** the hard 1022±80 gate is the
   strongest energy constraint expressible on this surface; it removed only
   18 of 117 FP while removing 30 TP. No softer/smarter variant can remove
   more bg than the hard gate's 18 — and every TP it spends is worth ~6× an
   FP at this class balance.
2. **All four granularities null on bg beyond iter 2's −1 FP:** subset
   total (iters 1-4, 7-8), best bipartition split (iter 5), max-hit cap
   (iter 6), winning-pair chains (iter 9). FP stayed within 114-117
   throughout; the seed list (#1-#5) is exhausted.
3. **Monotone closure:** every penalty weaker than a null penalty is null
   (one-sided ⊂ two-sided, iter 8); every penalty stronger than iter 2
   costs 4-30 TP (iters 1, 3-7, 9) — always the same fragile sub-1022
   partial-deposit signal population.
4. **Physical conclusion:** the residual ~116 FP are energy-indistinguishable
   from signal at every granularity tested — consistent with activation
   background containing *real* β⁺/511-pair events. The angle×energy joint
   constraint cannot separate real annihilation in the background source
   from real annihilation in the signal source.

**Proposed V4 directions (out of V3 scope, logged per protocol):**
- *Spatial origin / vertex coherence:* signal annihilations originate in the
  source volume; activation-bg annihilations originate in detector/passive
  material. The back-to-back pair's line-of-response should pass near the
  source for signal — a geometric lever V3's surface (energies only) could
  not express. Requires positions+geometry coupling downstream or a new
  feature.
- *Event-level coherence aggregator:* combine the three features' WINNING
  subsets (same-subset consistency across delta_E / angle / arm) instead of
  independent per-feature mins — requires touching the classifier, outside
  `annihilation_angle`.

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
