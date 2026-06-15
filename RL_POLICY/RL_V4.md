# RL Policy V4 — Replace `annihilation_angle` with the blind LOR score

V1–V3 converged at **MCC = 0.8792** (V3 iter 2, commit `342e417`). The
convergence is structural: `annihilation_angle` scans inter-hit difference
vectors for an anti-parallel pair, which (a) never measures actual photon
directions (the vertex is not a hit), (b) is not invariant under hit index
relabeling, (c) saturates combinatorially at high multiplicity, and (d) is
NaN on 2-hit events — the cleanest annihilation topology. V2/V3 optimized
*around* these defects and exhausted the reachable space.

V4 **replaces the feature outright** with the blind line-of-response
reconstruction (`physics/annihilation_lor.py`, commit `3addd23`): bipartition
hits into two photon chains, the line between first hits supplies each
chain's incoming direction, hypotheses are graded on 511 keV per-chain
energy residuals + first-scatter Compton consistency (E_in = 511 known).
`annihilation_angle` is not edited, tuned, or kept as a fallback — it goes.
(The function stays in `physics_factors.py` as dead code during the loop to
bound the diff; deleting it is post-loop cleanup.)

**Truth-harness facts already established** (tests_specialized/
test_annihilation_lor.py, 10 s for all 52,931 events — do NOT re-derive
these with pipeline evals):

- Topology census: 9.9% of signal events are pair-complete; 90.1% have
  exactly one fully absorbed photon. The one-photon case is the MAIN case.
- Blind axis vs ANNI truth: median 9.8° (random = 60°), 41% within 5°.
- Score class overlap TV = 0.271 vs legacy `annihilation_angle` TV = 0.113.
- σ_deficit trade-off (selection): 30 → TV .284 / axis 12.5°; 250 → TV .188
  / axis 9.8°. Decoupling (asym select / sym score) → TV .271 / axis 9.8°.
- NaN population: score is NaN for n_hits < 2 or > 12 — 43 signal events,
  28,077 bg events (mostly 1-hit).

**Trigger:** when the user says `Execute RL_V4.md` (or "start V4", "run the
V4 loop", "go"), begin at Phase 0. Until then, this is a normal doc.

---

## Mission

Maximize eval MCC by replacing the `annihilation_angle` feature with
`annihilation_lor_score` and tuning the new feature + its integration into
the Bayesian model. **Success bar: beat MCC 0.8792 (the V1–V3 champion).**

Guardrail: recall ≥ 0.9036 (V3 baseline recall 0.9536 − 0.05). `run.py`
must complete end-to-end.

## Baseline

The bar is the standing best, **MCC = 0.8792 / recall 0.9536 / FP 116**
(current `rl-cli` HEAD physics). Phase 0 is NOT behavior-preserving — it is
the replacement itself. Its result is logged as iter 0 and compared against
the 0.8792 bar; the scoreboard's "best" starts at 0.8792, so nothing is
committed until V4 actually beats the V3 champion.

## Phase 0 — the swap (iter 0 only)

Three named edits, nothing else:

1. **`dataset/datasets.py` (call site):** replace the
   `annihilation_angle(...)` call with `annihilation_lor_score(raw_positions,
   raw_energies)` computed from the event's RAW hits (extract per-hit
   positions/energies from `event_sim` the same way `event_data_processing`
   does at its top — the LOR consumes raw hits, not the subset pool).
   Keep the existing tensor/variable names (`*_annihilation_angle` slots now
   carry the LOR score) to bound the diff; renaming is post-loop cleanup.
2. **`pipeline/train.py` (binning):** the ΔE×angle matrix currently uses
   `spacing_y="linear"` (calibrated for a cosine in [-1, 1]). The LOR score
   is residual-like (≥ 0, point mass near 0, long tail) — switch that axis
   to `spacing_y="log"` with `log_y_floor=1e-3`, mirroring the arm axis.
3. Run `run.py` end-to-end. Log iter 0 with full metrics + harness columns.

Phase 0 deliverable: one scoreboard row. Commit/push only if iter 0 already
beats 0.8792 (`phase0: swap annihilation_angle -> LOR score`). From iter 1
the writable surface locks down per the constraints below.

If the swap requires touching any file beyond the two named ones, stop and
report — do not improvise.

## Phase 1 — iteration loop

Same per-iter protocol as V1–V3: read state, one hypothesis, edit, evaluate,
decide, log. Commit/push only on a new best (> 0.8792 and > prior V4 best).

**Harness pre-flight (new in V4):** before every pipeline eval, run
`pytest tests/tests_specialized/test_annihilation_lor.py --no-cov -q`
(~10 s, pure Python). All 3 tests must pass; record the run's TV and axis
median in the scoreboard row. If the harness fails, the hypothesis is
rejected before spending a pipeline eval. The harness is a *gate and
diagnostic*, not the objective — MCC decides keep/revert.

### Action space — the LOR feature and its integration

Seed hypotheses, ranked. One verb per iter.

1. **σ_deficit on MCC** (`SIGMA_E_DEFICIT_KEV` ∈ {30, 100, 500}; 250 is the
   Phase 0 default). The harness mapped the TV/axis trade; MCC arbitrates
   which the classifier prefers.
2. **Geometry-only reported score**: drop the symmetric energy term from
   the returned `score` (keep it for selection). Makes the feature maximally
   orthogonal to `delta_E` — marginal TV will fall, but the conditional
   P(lor|ΔE) may carry MORE new information. This is the cleanest test of
   the redundancy question.
3. **NaN → sentinel**: map n_hits < 2 to a large finite score instead of
   NaN. Currently 28,077 bg vs 43 signal events contribute nothing through
   this feature (NaN → density 0 both classes → ratio 1). A sentinel makes
   "cannot be two photons" itself evidence. Watch recall: costs ≤ 43 TP
   worst case.
4. **Single-ΔE-evidence factorization**: set `modeling.double_count_deltaE:
   false` in `configs/config.yaml` (the flag exists for exactly this A/B).
   One-line hypothesis; affects arm too — evaluate on MCC like any other.
5. **σ_cos sweep** (`SIGMA_COS` ∈ {0.02, 0.1}; 0.05 default) — geometry
   residual scale vs the Doppler/position-resolution floor.
6. **σ_E sweep on the reported symmetric residual** (`SIGMA_E_KEV` ∈
   {10, 80}; 30 default).
7. **Cone fallback for the one-complete-photon majority**: when the best
   hypothesis's secondary chain has < 2 hits (no LOR cross-check), grade
   the primary chain's internal Compton consistency instead. Validate on
   the harness (axis + TV) before the pipeline eval.
8. **n > 12 events**: greedy energy-balanced bipartition instead of NaN
   (bounded cost), recovering the multiplicity tail.

### Anti-repeat

V1–V3 scoreboards are about the old feature's candidate pool — structurally
moot now. What transfers:

- **Density re-fit non-monotonicity** (V1 iter 6): every feature change
  reshapes the train-time densities; small MCC deltas (±0.003) are noise —
  do not chase them.
- **MCC is TP-sensitive** at this class balance (~6:1 FP-worth per TP):
  recall-cheap hypotheses first.
- **Gentle beats hard** (V3): prefer re-ranking/soft penalties over
  drops/gates inside the feature.
- Do not re-run harness-established sweeps as pipeline iterations without a
  reason MCC would disagree with TV (hypotheses 1–2 ARE such reasons; a
  blind third σ_deficit re-sweep is not).

### Stop conditions

- 50 Phase 1 iterations completed.
- 10 consecutive Phase 1 iterations without a new best.
- 3 consecutive iterations where `run.py` errors out (re-run once before
  counting — ROOT teardown segfaults are known flakes).
- MCC ≥ 0.95.
- Recall floor breach (< 0.9036) for two consecutive iters.

**Failure path:** if the loop stops without beating 0.8792, revert
`datasets.py`/`train.py`/`config.yaml` to the V3 champion state, write the
halt summary, and propose V5 = add the LOR score as a FOURTH feature
alongside `annihilation_angle` instead of replacing it (out of V4 scope).

## Scoreboard

Fresh `SCOREBOARD_V4.md` at repo root. V1–V3 schema plus two columns:
**harness TV** and **harness axis median (deg)** from the pre-flight run.
Iter 0 = the post-swap measurement.

## Constraints (hard)

1. **Read access:** entire repository.
2. **Write access:**
   - Phase 0 (iter 0 only): `dataset/datasets.py` (the one call site) and
     `pipeline/train.py` (the one spacing line).
   - Phase 1 (iter 1+): `src/betadecay-analysis/physics/annihilation_lor.py`
     (constants, helpers, function bodies). Additionally, a single
     hypothesis may touch `configs/config.yaml` (`double_count_deltaE`
     only, seed #4) or revisit the Phase-0 lines in `datasets.py` /
     `train.py` when the hypothesis is literally about those lines (e.g.
     sentinel handling, binning floor). Nothing else.
3. Do not change `annihilation_lor`'s public conventions: score is a
   scalar ≥ 0, lower = more annihilation-like, NaN = untestable. Downstream
   log-binning depends on this.
4. `event_processing.py`, `physics_factors.py` (`arm`, `delta_E`, the dead
   `annihilation_angle`), `modeling/`, `mathematics/` are **frozen**.
5. No new dependencies. No revan / `.tra` reads in the feature path.
6. pytest: ONLY `tests/tests_specialized/test_annihilation_lor.py` (the
   pre-flight). Do not run the full suite; do not edit any test. If a
   feature change legitimately shifts harness expectations, stop and
   report instead of editing the test.
7. The ANNI truth (IA records) must never feed the runtime feature — it is
   harness-only. Any hypothesis that reads IA lines inside
   `annihilation_lor.py` or `datasets.py` is invalid by construction.

## Idea backlog

Read `EVENT_RECONSTRUCTION_THEORY.md` §2 (CKD), §7 (two-photon) and the
module docstring of `annihilation_lor.py` at the start of every iteration.
Logged for V5+, out of V4 scope: source-volume prior on the LOR line
(needs source geometry plumbed in), event-level same-subset coherence
across features, four-feature model (LOR + legacy angle).

## Anti-patterns

All V1–V3 anti-patterns apply. Additionally:

- Do not optimize the harness metrics (TV, axis) as the objective — they
  gate and explain; MCC decides. A hypothesis that raises TV but drops MCC
  is a revert.
- Do not bundle the swap with a tuning change in Phase 0. Iter 0 is the
  pure swap; attribution requires it.
- Do not resurrect `annihilation_angle` mid-loop "just to compare" — that
  is V5's four-feature question, not V4's.
- Do not let a feature edit silently change the NaN population (the
  pre-flight census + NaN counts will catch it; check them per iter).
