# RL Policy V5 — Four-feature model: LOR score alongside the legacy angle

V4 replaced `annihilation_angle` with the blind LOR score and fell 3 eval FP
short of the champion (best 0.8770 vs bar 0.8792, identical TP/FN). Per the
V4 failure path, the pipeline was reverted to the V3 champion and
`physics/annihilation_lor.py` was left at its harness-validated iter-9 state
(geometry-only score, σ_cos = 0.02, selection-side internal-chain grading;
TV 0.387, axis median 8.8°). V5 wires that score in as a **FOURTH feature
next to the legacy three**, instead of replacing anything.

**Why this should beat the bar when replacement didn't:** the LOR-only
classifier reached within 3 FP of the champion *without* the legacy angle,
and the two features grade different things (legacy: subset-pool vector
geometry; LOR: raw-hit chain kinematics + internal Compton consistency).
The Bayesian layer can hold both and let the densities decide per event.

**Label-edge diagnostic (2026-06-10, full-file harness) — read before
hypothesizing about the residual FP:**

- The "remaining FPs are real annihilations just outside the ±2.87 keV
  label tolerance" hypothesis is **REFUTED**. Only 97 of 48,453 bg events
  contain an annihilation missed by <10 keV, and their LOR scores are HIGH
  (median 488 vs ordinary-bg 18) — the internal-chain grading *sees* their
  incompleteness. The iter-7–9 work closed the very loophole V3's halt
  summary worried about. There is no proven ceiling at 0.8792.
- The bg population the LOR score finds most annihilation-like (score below
  the signal 10th percentile, n≈8.8k) is **73% events containing NO
  annihilation at all** — geometric coincidences concentrated in degenerate
  low-multiplicity topologies (the dof = 0 zero-score mass: 41.6% of finite
  bg vs 6.7% of signal).
- The iter-9 score is **non-monotone in aggregate** (bg median 17.6 <
  signal median 152.3, because bg piles at exactly 0 and in the far tail).
  The density-ratio model prices this correctly (the zero bin is ~6:1
  bg-ward evidence); never apply "lower = more signal-like" threshold
  reasoning to this feature, and keep the zero mass inside its own bin.

**Trigger:** when the user says `Execute RL_V5.md` (or "start V5", "run the
V5 loop", "go"), begin at Phase 0. Until then, this is a normal doc.

---

## Mission

Maximize eval MCC by adding `annihilation_lor_score` (iter-9 module state,
unmodified at iter 0) as a fourth classifier feature alongside `delta_E`,
`annihilation_angle`, and `arm`. **Success bar: beat MCC 0.8792.**

Guardrail: recall ≥ 0.9036. `run.py` must complete end-to-end.

## Baseline

The V3 champion at current `rl-cli` HEAD: **MCC 0.8792, recall 0.9536,
575 / 116 / 28 / 5876**. Scoreboard "best" starts at 0.8792; commit/push
only on beating it.

## Phase 0 — wire the fourth feature (iter 0 only)

The named edits, nothing else:

1. **`dataset/datasets.py`:** keep the legacy `annihilation_angle` exactly
   as is; ADD `_lor = annihilation_lor_score(raw_positions, raw_energies)`
   computed from the event's raw hits, with parallel
   `bdecay/bg/gen/combined` lor tensors threaded through the return.
2. **`pipeline/train.py`:** add the ΔE×lor density pair, mirroring the
   ΔE×arm block: `spacing_x="log"` (`log_x_floor=0.1`), `spacing_y="log"`,
   `log_y_floor=1e-3` (the score's zero point-mass lands in the bottom bin
   — that bin IS the 6:1 evidence, keep it).
3. **`pipeline/eval.py` / lookups:** per-event `p_beta_lor`, `p_bg_lor`
   from the new matrices (the generic lookup helpers need no change).
4. **`modeling/calculate_probablities.py`:** extend `R` with the lor term,
   same form as angle/arm under the current `double_count_deltaE: true`
   (i.e. multiply by the joint-derived `(p_beta_lor+eps)/(p_bg_lor+eps)`).
5. **`run.py`:** thread the new tensors through the existing call shape.

NaN convention unchanged: NaN lor (n<2 or n>12) → density 0 both classes →
term ratio 1 → the other three features decide. (V4 iter 4 proved
re-pricing the NaN population corrupts the shared normalization — do not
revisit.)

Run `run.py`; log iter 0 with full metrics + harness columns. Commit only
if it already beats the bar (`phase0: add LOR score as fourth feature`).

If the wiring forces a change beyond the five named files, stop and report.

## Phase 1 — iteration loop

Per-iter protocol as V1–V4. Harness pre-flight per V4 (all 3 tests pass;
TV + axis median logged — **TV is a gate, never a ranking**: four
documented TV/MCC inversions in V4).

### Action space, ranked

1. **lor-axis binning:** `log_y_floor` ∈ {1e-4, 1e-2} (where the zero mass
   bin boundary sits is the single most loaded discretization choice);
   optionally `n_bins` for the lor matrix only.
2. **Conditional form for the lor term:** marginal `P(lor)` instead of the
   ΔE-conditioned joint. V4 iter 5 showed 25-bin conditionals are sparse
   (≤603 signal train events over 625 cells); the lor score is already
   ΔE-orthogonal by construction (V4 iter 3), so the marginal may be both
   denser and unbiased. This is V5's sharpest model-side hypothesis.
3. **Feature-side, from the V4 halt list:** joint selection of (first_a,
   first_b) WITH internal residuals (iter 9 grades internals only at the
   first-scatter argmin); n>12 greedy energy-balanced bipartition (V4 seed
   #8, never tried — the multiplicity tail).
4. **Degenerate-mass split:** give dof = 0 hypotheses a distinct sentinel
   *within the finite range* (e.g. exactly 0 vs epsilon-offset graded
   scores) so the zero bin contains only true degenerates. Feature-side
   relabeling, not density re-pricing — V4 iter 4's failure mode does not
   apply, but verify the harness census stays unchanged.
5. **Targeted no-ANNI coincidence levers** (from the label-edge
   diagnostic): the annihilation-like bg is 73% no-ANNI geometric
   degenerates — hypotheses that specifically raise THEIR score (e.g.
   minimum-lever-arm weighting on the LOR, sub-cm chains carry no direction
   information) without touching real-annihilation geometry.

### Anti-repeat (V4 axes closed — do not re-open)

- σ_deficit (250 optimal: tighter hurts, looser inert), σ_cos (0.02 kept,
  coarser is MCC-inert), χ²/dof normalization (multiplicity trend in the
  raw sum is informative), NaN→sentinel at the call site,
  `double_count_deltaE: false`, symmetric-energy term in the report.
- All V1–V3 lessons transfer (density re-fit non-monotonicity, ~6:1
  TP-worth, gentle-beats-hard).

### Stop conditions

- 50 Phase 1 iterations; 10 consecutive without a new best; 3 consecutive
  `run.py` errors (re-run once first); MCC ≥ 0.95; recall < 0.9036 twice
  consecutively.

**Failure path:** revert all Phase 0 wiring (pipeline back to the V3
champion), keep `annihilation_lor.py` as is, write the halt summary. V6
candidates to log, in order: (a) per-event FP characterization on the
Linux machine (dump the eval split's FP event IDs + per-feature likelihood
terms — V5 runs blind on WHICH events resist; that tooling ends the
guessing), (b) conditional-on-ΔE separability metric in the harness so the
pre-flight can rank, (c) source-volume prior on the LOR — ranked LAST now:
the label is detection-completeness, not origin, and the label-edge
diagnostic already shows the dangerous bg is no-ANNI coincidences, which a
source prior does not address.

## Scoreboard

Fresh `SCOREBOARD_V5.md` at repo root, V4 schema (incl. harness TV + axis
columns). Iter 0 = the four-feature wiring measurement.

## Constraints (hard)

1. **Read access:** entire repository.
2. **Write access:** Phase 0 (iter 0 only): the five named files, only for
   the named additions. Phase 1: `physics/annihilation_lor.py` + the
   Phase-0 lines themselves when the hypothesis is literally about them
   (binning, term form). Nothing else.
3. **`annihilation_angle` (legacy), `arm`, `delta_E`, `event_processing.py`
   are frozen.** V5 adds; it does not retune the existing three features.
4. LOR score conventions unchanged: scalar ≥ 0, NaN = untestable, zero
   point-mass preserved. Truth (IA records) never feeds the runtime path.
5. No new dependencies. pytest: only
   `tests/tests_specialized/test_annihilation_lor.py`. Do not edit tests.
6. One verb per iter; commit/push only on a new best (> 0.8792).

## Anti-patterns

All V1–V4 anti-patterns apply. Additionally:

- Do not treat harness TV as a ranking (four V4 inversions). Gate + axis +
  census only.
- Do not "simplify" by removing the legacy angle mid-loop — that was V4,
  it lost by 3 FP, and the four-feature premise is exactly that both
  carry non-identical information.
- Do not re-price NaN/degenerate populations at the density layer (V4
  iter 4). Feature-side relabeling within the finite range (seed #4) is
  the only sanctioned form.
