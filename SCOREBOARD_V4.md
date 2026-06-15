# Scoreboard V4 — Replace `annihilation_angle` with the blind LOR score

**HALTED 2026-06-10 — bar (MCC 0.8792) not beaten; V4 best: iter 9,
MCC=0.8770 (geometry-only score + selection-side internal-chain grading).
Pipeline reverted to the V3 champion per the failure path; see halt summary
below the table. `annihilation_lor.py` retains the iter-9 feature state
(unwired) for V5.**

Loop surface: Phase 0 (iter 0) = the swap itself (`dataset/datasets.py` call
site + `pipeline/train.py` ΔE×angle y-axis binning linear→log). Phase 1
(iter ≥1) = `physics/annihilation_lor.py` (constants, helpers, bodies), plus
the single-hypothesis exceptions named in `RL_POLICY/RL_V4.md` §Constraints.

**Success bar: beat MCC 0.8792 (V1–V3 champion, commit `342e417`).** The
scoreboard "best" starts at 0.8792 — nothing is committed until V4 beats it.

**Baseline recall = 0.9536. Guardrail floor = 0.9036** (recall ≥ baseline − 0.05).
Stop if recall < 0.9036 for two consecutive iters.

MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)), computed from the
eval confusion matrix printed by `predict`.

**Harness pre-flight** (per iter, before the pipeline eval):
`pytest tests/tests_specialized/test_annihilation_lor.py --no-cov -q` — all 3
must pass; TV and axis median recorded below. Gate + diagnostic only; MCC
decides keep/revert.

| Run ID | Timestamp (UTC) | Status | Hypothesis | What changed | Harness TV | Harness axis med (deg) | MCC | Δ vs best | F1 | Precision | Recall | FPR | TP / FP / FN / TN | Notes / next idea |
|--------|-----------------|--------|------------|--------------|-----------|------------------------|-----|-----------|----|-----------|--------|-----|--------------------|-------------------|
| 0 | 2026-06-10T04:30:00Z | below bar | Phase 0 pure swap (no tuning): the LOR score replaces `annihilation_angle` outright | `datasets.py`: call site now `annihilation_lor_score(raw_positions, raw_energies)` on RAW hits (extracted like `event_data_processing`'s top); `train.py`: ΔE×angle y-axis `spacing_y` linear→log, `log_y_floor=1e-3` | 0.271 | 11.3 | 0.8712 | -0.0080 | 0.8814 | 0.8230 | 0.9486 | 0.0205 | 572 / 123 / 31 / 5869 | **Below the V3 champion (0.8792): TP −3, FP +7 vs 575/116.** Notable asymmetry: TRAIN recall went UP (0.9573→0.9635) while eval recall fell (0.9536→0.9486) — the higher-TV feature (0.271 vs legacy 0.113) is being partially spent on train-fit, consistent with V1's density re-fit non-monotonicity. The swap itself is sound (harness TV 2.4× the legacy feature; NaN census unchanged at 43 sig / 28,077 bg). Defaults were chosen on harness TV, not MCC — tuning starts now. Next (iter 1): seed #1 σ_deficit sweep, SIGMA_E_DEFICIT_KEV 250→30 (harness says TV rises .271→.284 with axis 9.8→12.5°; MCC arbitrates whether the classifier prefers the tighter selection). |
| 1 | 2026-06-10T04:50:00Z | reverted | seed #1 σ_deficit sweep, point 1: SIGMA_E_DEFICIT_KEV 250→30 (selection ≈ symmetric; harness-predicted TV gain) | `SIGMA_E_DEFICIT_KEV = 30.0` (single constant) | 0.284 | 11.3 | 0.8702 | -0.0090 | 0.8805 | 0.8228 | 0.9469 | 0.0205 | 571 / 123 / 32 / 5869 | **WORSE than iter 0: TP −1, FP ±0.** The harness TV gain (.271→.284) did NOT translate to MCC — first direct confirmation in V4 that marginal TV and the classifier disagree (the conditional P(lor\|ΔE) is what matters). Tighter selection mis-picks hypotheses on partial-deposit signal (axis unchanged at 11.3° but selection now punishes the 90% one-photon majority). Reverted. Next (iter 2): seed #1 point 2, the opposite direction — SIGMA_E_DEFICIT_KEV 250→500 (even more lenient deficits; helps the one-photon majority's hypothesis selection). |
| 2 | 2026-06-10T05:10:00Z | reverted | seed #1 σ_deficit sweep, point 2: SIGMA_E_DEFICIT_KEV 250→500 (more lenient secondary-chain deficits) | `SIGMA_E_DEFICIT_KEV = 500.0` (single constant) | 0.271 | 11.3 | 0.8712 | -0.0080 | 0.8814 | 0.8230 | 0.9486 | 0.0205 | 572 / 123 / 31 / 5869 | **Confusion matrix IDENTICAL to iter 0 — σ_deficit axis CLOSED.** Beyond 250 the selection is already effectively deficit-blind (the harness agreed: TV/axis unchanged); below 250 it mis-picks on the one-photon majority (iter 1). 100 is dominated (between two losers). Reverted to 250. Next (iter 3): seed #2 geometry-only reported score — return `geo_resid` alone (energy terms still drive selection). Marginal TV will fall, but the model conditions on ΔE; this is the cleanest test of whether the LOR's energy term is redundant with delta_E. |
| 3 | 2026-06-10T05:25:00Z | **kept (working state)** | seed #2 geometry-only reported score: return `geo_resid` alone; energy residuals still drive hypothesis SELECTION | `annihilation_lor`: `score = geo_resid` (symmetric energy term dropped from the report; `_symmetric_energy_residual` now unused) | 0.462 | 11.3 | 0.8713 | -0.0079 | 0.8817 | 0.8261 | 0.9453 | 0.0200 | 570 / 120 / 33 / 5872 | **MCC ≈ iter 0 (+0.0001, noise) but the prediction inverted: marginal TV ROSE .271→.462** — the geometry residual is the discriminating part; the energy term diluted it AND was redundant with ΔE (dropping it cost zero MCC). FP −3 / TP −2 vs iter 0. Kept as working state: same MCC, structurally cleaner (feature now fully orthogonal to ΔE), and seed #7 (cone fallback) composes naturally with a pure-geometry score. Next (iter 4): seed #3 NaN→sentinel — map n_hits<2 (28,077 bg / 43 sig events, currently ratio-1 passthrough) to a large finite score at the `datasets.py` call site (constraint 2 exception: sentinel handling is literally that line). "Cannot be two photons" becomes evidence. Recall risk ≤ ~9 eval TP worst case. |
| 4 | 2026-06-10T05:50:00Z | reverted | seed #3 NaN→sentinel: n_hits<2 events (28,077 bg / 43 sig) get score 1e4 (a decade above the real max ~800) instead of NaN, at the `datasets.py` call site | `datasets.py`: `if n_hits < 2: _annihilation_angle = 1e4` (feature module untouched; harness TV/axis carried from iter 3) | 0.462 | 11.3 | 0.8626 | -0.0166 | 0.8725 | 0.8006 | 0.9585 | 0.0240 | 578 / 144 / 25 / 5848 | **WORSE, and not via the predicted mechanism: TP +8 (recall 0.9585, best ever) but FP +24.** The predicted ≤9-TP recall risk never materialized — instead, parking 53% of bg's mass in the sentinel bin RENORMALIZED bg's density downward across all real-score bins, shifting every finite-score event signal-ward: +8 TP, +24 FP (~3:1 against at this balance). The sentinel as *evidence* works; the side effect on the shared density normalization is the killer. Reverted. Next (iter 5): seed #4 single-ΔE-evidence factorization — `modeling.double_count_deltaE: false` (config one-liner, the sanctioned A/B). Newly live question: with the score now ΔE-orthogonal (iter 3), down-weighting ΔE (used twice today) should cost the LOR axis nothing and may rebalance the 116-FP-class events that are 511-like in ΔE. |
| 5 | 2026-06-10T06:10:00Z | reverted | seed #4 single-ΔE-evidence factorization: P(ΔE)·P(angle\|ΔE)·P(ARM\|ΔE) instead of the double-counted joints | `configs/config.yaml`: `modeling.double_count_deltaE: false` (one line) | 0.462 | 11.3 | 0.8542 | -0.0250 | 0.8665 | 0.8146 | 0.9254 | 0.0212 | 558 / 127 / 45 / 5865 | **WORSE on both sides: TP −12, FP +7 vs iter 3.** Double-counting ΔE is load-bearing, not a bug: at 25 bins the conditionals P(·\|ΔE) are sparse (≤603 signal train events spread over 625 cells), so the joint's implicit second ΔE vote is what carries the energy evidence. Factorization A/B closed. Reverted to `true`. Next (iter 6): seed #5 σ_cos sweep, point 1: SIGMA_COS 0.05→0.02 — with the geometry-only score this is now the master constant: tightens selection's geometry/energy balance AND inflates scores ×6.25, pulling the signal point-mass at 0 out of the 1e-3 log-binning floor (finer near-zero resolution). |
| 6 | 2026-06-10T06:35:00Z | **kept (working state)** | seed #5 σ_cos sweep, point 1: SIGMA_COS 0.05→0.02 (geometry dominates selection; score scale ×6.25) | `SIGMA_COS = 0.02` (single constant) | 0.480 | 9.0 | 0.8713 | -0.0079 | 0.8817 | 0.8261 | 0.9453 | 0.0200 | 570 / 120 / 33 / 5872 | **Confusion matrix IDENTICAL to iter 3 — MCC-inert** (monotone-ish rescale; adaptive log bins absorbed it), but the harness improved on BOTH axes: TV .462→.480, axis median 11.3°→9.0° (tighter geometry weighting picks better hypotheses). Kept: zero MCC cost, better selection platform for the fallback work. σ_cos=0.1 (other direction) skipped — the axis is demonstrably MCC-inert, coarser would at best tie (anti-repeat). Next (iter 7): seed #7 cone fallback — the score currently grades ONLY each chain's first scatter; internal scatters are never checked. For 1-hit-secondary hypotheses (the one-complete-photon majority), add the ≥3-hit primary chain's internal Compton-consistency residuals (greedy ordering continuation, 511-anchored energy ladder) to the reported score; selection untouched. Harness-validate (axis must hold ~9.0°) before the pipeline eval. |
| 7 | 2026-06-10T07:00:00Z | **kept — V4 working best** | seed #7 cone fallback: for 1-hit-secondary hypotheses, add the ≥3-hit primary chain's internal Compton-consistency residuals (greedy walk, 511-anchored energy ladder) to the reported score; selection untouched | new `_internal_chain_residual` helper; `annihilation_lor` adds it for the one-complete-photon case | 0.377 | 9.0 | 0.8725 | -0.0067 | 0.8824 | 0.8223 | 0.9519 | 0.0207 | 574 / 124 / 29 / 5868 | **V4 working best (+0.0012 vs iter 3/6): TP +4, FP +4 — net positive at this balance.** Third TV/MCC inversion: harness TV fell .480→.377 (internal residuals push signal's own distribution up — Doppler + greedy mis-orderings) yet MCC ROSE; marginal TV is officially a poor proxy on this feature. Axis held at 9.0° (selection untouched), census unchanged. Gap to bar: ~8 FP at constant TP. Next (iter 8): extend the winner — grade internal residuals for BOTH ≥3-hit chains in every hypothesis (drop the 1-hit-secondary scoping); the pair-complete 9.9% currently gets no internal check at all. |
| 8 | 2026-06-10T07:25:00Z | **kept — V4 working best** | extend iter 7: internal Compton-consistency residuals for BOTH ≥3-hit chains in the reported score (1-hit-secondary scoping dropped) | `annihilation_lor`: both-chain `_internal_chain_residual` accumulation (post-selection) | 0.360 | 9.0 | 0.8763 | -0.0029 | 0.8860 | 0.8273 | 0.9536 | 0.0200 | 575 / 120 / 28 / 5872 | **+0.0038 vs iter 7 (beyond the ±0.003 noise band): TP +1, FP −4, recall fully restored to the V3-baseline 0.9536.** The bar is now exactly 4 FP away at identical TP (V3 champion 575/116). TV fell again (.377→.360) while MCC rose — internal grading is doing real work that marginal TV cannot see. Next (iter 9): move internal grading INTO selection (evaluated at each bipartition's first-scatter argmin; only adds to the residual so the energy prune stays a valid lower bound) — misordered partitions currently win on energy+first-scatter and only then get charged; letting internals vote should pick better partitions outright. |
| 9 | 2026-06-10T08:00:00Z | **kept — V4 working best** | selection-side internal grading: internal residuals accumulate inside the bipartition loop (at each partition's first-scatter argmin) and vote on the winner | `annihilation_lor` loop: internal walk per surviving bipartition; stored residual carries it; post-loop duplication removed | 0.387 | 8.8 | 0.8770 | -0.0022 | 0.8867 | 0.8285 | 0.9536 | 0.0199 | 575 / 119 / 28 / 5873 | **FP −1 at constant TP (+0.0007, directionally consistent with iters 7-8).** Axis median improved to 8.8° (best) — internals do pick better partitions. Harness runtime 13s→60s (prune keeps it tractable). Bar is 3 FP away. 9th consecutive iter below bar — ONE left before the stop. Zero-bin diagnostic: score<1e-3 is 41.6% of finite bg vs 6.7% of signal (ratio 0.16, ~6:1 bg-ward) — the degenerate 2-hit mass is already optimally exploited; n≥3 separation is TV 0.181 with multiplicity smeared into the raw sum. Next (iter 10, last shot): χ²/dof — normalize the reported score by graded-scatter count (dof=0 stays exactly 0; selection stays raw). |
| 10 | 2026-06-10T08:40:00Z | reverted | χ²/dof report: normalize the reported geometry score by the number of graded scatters (dof=0 partitions stay exactly 0; selection stays on raw sums) | `annihilation_lor`: post-loop `score = geo_resid / dof` with dof from the winning partition's chain sizes | 0.437 | 8.8 | 0.8746 | -0.0046 | 0.8844 | 0.8259 | 0.9519 | 0.0202 | 574 / 121 / 29 / 5871 | **WORSE: TP −1, FP +2 vs iter 9 — and the FOURTH TV/MCC inversion (TV .387→.437, MCC down).** The multiplicity trend in the raw sum is apparently informative, not nuisance: high-multiplicity bg accumulates residual mass that per-dof normalization forgives. Reverted to iter-9 state. **STOP CONDITION: 10 consecutive Phase-1 iterations without a new best (> 0.8792). Loop halted; failure path executed (see halt summary).** |

---

## V4 halt summary (2026-06-10, after iter 10)

**Bar not beaten. V4 best: iter 9, MCC = 0.8770** (geometry-only LOR score,
σ_cos = 0.02, selection-side internal-chain grading) vs the V1–V3 champion
**0.8792** (commit `342e417`) — short by 0.0022 ≈ 3 eval FP at identical
TP/FN (V4: 575/119/28/5873 vs V3: 575/116/28/5876). Nothing was committed
during the loop; per the failure path, `datasets.py` / `train.py` /
`config.yaml` are reverted to the V3 champion state (verified by a full
`run.py` eval reproducing 575/116/28/5876, MCC 0.8792) and
`physics/annihilation_lor.py` retains the iter-9 state — harness-validated
(TV 0.387, axis median 8.8°, 3/3 tests) but unwired.

### What the loop established

1. **The replacement closed 90% of its own gap but not the last 3 FP.** The
   pure swap (iter 0) opened at 0.8712; three structural wins (geometry-only
   score +0.0001, both-chain internal grading +0.0050, selection-side
   grading +0.0007) recovered recall to exactly the V3 baseline 0.9536 and
   FP to 119 vs the champion's 116.
2. **Marginal TV is a broken proxy for MCC on this feature — four documented
   inversions** (iters 1, 7, 8, 10). The harness TV ranged 0.271→0.480
   across iterations with essentially zero correlation to eval MCC. The
   conditional structure P(score | ΔE) is what the classifier consumes;
   future loops should treat TV only as a sanity floor, never a ranking.
3. **The energy half of the LOR score is fully redundant with ΔE** (iter 3:
   dropping it cost nothing) — but its *selection* role is load-bearing and
   already optimal at σ_deficit = 250 (iters 1–2: tighter hurts, looser is
   inert).
4. **Internal-chain Compton grading is the real discovery of V4** (+0.0057
   cumulative, iters 7–9): the original first-scatter-only score left every
   scatter beyond the second ungraded; walking the full chain both in the
   report and in selection improved MCC monotonically AND the truth-graded
   axis (9.8°→8.8° median). This transfers directly to V5.
5. **Degenerate populations are already optimally priced by the density
   model:** the 2-hit zero-score mass is 6:1 bg-ward evidence as-is
   (likelihood ratio 0.16), and the NaN→sentinel experiment (iter 4)
   showed that re-pricing the 1-hit population corrupts the shared density
   normalization (FP +24 for TP +8). The Bayesian layer, not the feature,
   owns these populations.
6. **Model-layer levers are closed:** single-ΔE factorization (iter 5)
   starves the sparse 25-bin conditionals (−0.0250); the double-counted
   joint is structurally load-bearing at this sample size.

### Why the last 3 FP are out of reach on this surface

The residual eval FP are background events whose hits admit a bipartition
with Compton-consistent chains — geometrically indistinguishable from true
annihilation by ANY blind per-event grade. V3 established they are also
energy-indistinguishable. The one lever that separates them physically —
the LOR axis of signal passes through the source volume; activation-bg
LORs originate in detector/passive material — is exactly the source-prior
idea the V4 policy logged as out of scope (needs source geometry plumbed
into the feature path).

### Proposed V5 (per the failure path, out of V4 scope)

**Four-feature model: add `annihilation_lor_score` (iter-9 state) as a
FOURTH feature alongside the legacy `annihilation_angle`,** rather than
replacing it. Rationale: V4 proved the LOR score carries real independent
signal (axis 8.8° vs truth; the V4-best classifier reaches within 3 FP of
champion *without* the legacy angle), and the two features disagree on
*which* events they grade well (legacy: subset-pool vector geometry; LOR:
raw-hit chain kinematics). The Bayesian layer would need a third joint
(or a P(lor | ΔE) conditional) — a `modeling/` + `train.py` change,
forbidden in V4. Secondary V5 candidates, in order: source-volume prior on
the LOR line (the physically-correct discriminator for activation bg);
selection-(i,k) chosen jointly with internal residuals (iter 9 grades only
at the first-scatter argmin); n>12 greedy bipartition (untried seed #8 —
the multiplicity tail).
