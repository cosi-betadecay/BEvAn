# RL Policy V2 — Clean-signal Reconstruction Loop

V1 (`RL_V1.md`) ran 10 iterations and halted. Best stayed at the baseline.
The post-mortem (see V1's scoreboard + repo audit) identified two reasons:

1. **Metric noise.** `modeling/calculate_probablities.py` counts NaN
   predictions as real confusion-matrix entries, and `pipeline/train.py`
   computes the class prior on `isfinite(...)` counts while applying it
   to all of eval. Together this makes the MCC delta of any filter
   contain an unknown NaN-reclassification + prior-shift component.
   Iter 6's "FP rose when I removed subsets" is the smoking gun:
   pure removal cannot create FPs, so something else is moving.

2. **Removal-only action space.** V1 could only delete subsets or
   reject events. Since features are `min`-aggregates over subsets,
   removal can only raise mins → make events look *less* signal-like.
   Signal/background argmin-subsets overlap in feature space, so every
   removal was near-zero-sum.

V2 fixes both. It runs in two phases.

**Trigger:** when the user says `Execute RL_V2.md` (or "start V2",
"run the V2 loop", "go"), begin at Phase 0. Until then, this is a
normal doc.

---

## Phase 0 — one-shot metric fix (iter 0 only)

Before any policy iteration, repair the evaluation signal. You are
permitted — **only on iter 0 of V2, only for these two specific
edits** — to modify:

- `src/betadecay-analysis/modeling/calculate_probablities.py`
- `src/betadecay-analysis/pipeline/train.py`

The two fixes:

1. **NaN mask in confusion matrix.** In `calculate_probablities.py`
   around lines 161–164, exclude entries where the prediction OR the
   ground-truth label is NaN before tallying TP/FP/FN/TN. Print a
   one-line `eval - NaN excluded: <n>` so the count is visible.
2. **Prior denominator consistency.** In `train.py:90-91`, either
   compute `n_beta_decay` / `n_bg` over the full population (not
   `isfinite(...)`), or apply the same `isfinite` mask at eval time
   when consuming the prior. Pick whichever requires the smaller diff;
   document the choice in the Phase 0 commit message.

After Phase 0:

1. Commit both files in one commit: `phase0: fix NaN tallying + prior denominator`.
2. Push to `rl-cli`.
3. Re-run `run.py` on the now-clean pipeline. This becomes the **V2
   baseline** (iter 0 of the V2 scoreboard).
4. **Lock the writable surface back down** for Phase 1. From iter 1
   onward, the only editable file is again
   `src/betadecay-analysis/physics/event_processing.py`.

If either fix changes a public function signature consumed elsewhere,
stop and report — do not improvise refactors.

## Phase 1 — iteration loop

Same mission as V1: maximize eval MCC by editing
`src/betadecay-analysis/physics/event_processing.py`. Same guardrails
(recall ≥ V2 baseline − 0.05; `run.py` must complete). Same per-iter
protocol: read state, one hypothesis, edit, evaluate, decide, log.
Same commit rule: push only on a new best.

What changes from V1:

### Expanded action space — subset *augmentation*, not just removal

V1 proved removal is near-Pareto-optimal on its own. V2 adds a second
verb: **synthesize new candidate subsets** alongside the enumerated
ones. Because features are `min` over subsets, *injecting* a better
candidate can lower a feature's min and pull signal further from the
boundary — the direction removal can't reach.

Concrete seed ideas, ranked:

1. **Back-to-back pair-subsets.** For each event, find hit pairs whose
   inter-vector cosine is ≤ −0.6 and energies each ≈ 511 keV; add
   them as explicit 2-hit candidate subsets. Targets `annihilation_angle`.
2. **Ordered Compton chains (CKD seed).** For size≥3 subsets, also
   emit the kinematically best-ordered permutation as its own
   candidate, so `arm`'s argmin can find the physically-ordered
   version instead of relying on lucky enumeration order.
3. **ΣE≈1022 pair-subsets.** Pairs whose summed energy is within
   ±50 keV of 1022; add as candidates for the two-photon feature.

Augmentation is allowed to coexist with light removal (e.g. keep the
iter-7 b2b-exempted clamp-artifact cut as a baseline filter once V2
baseline is set, *if* it pareto-improves on the clean signal — re-test it).

### Anti-repeat

Do not re-try any hypothesis already in V1's scoreboard (energy-window
removal in any form, lever-arm hard cut, n_hits event-level reject at
8/10/12). Phase 0 invalidates V1's *numbers*, but its *negative results
about removal* still hold qualitatively.

### Stop conditions (unchanged from V1, restated)

- 50 Phase 1 iterations completed.
- 10 consecutive Phase 1 iterations without a new best.
- 3 consecutive iterations where `run.py` errors out (re-run once
  before counting an error, per V1 iter 8 ROOT-teardown lesson).
- MCC ≥ 0.95.

## Scoreboard

Use a fresh `SCOREBOARD.md` at repo root for V2. Same schema as V1.
Iter 0 row is the **post-Phase-0 baseline**, not the original repo.
Keep V1's old scoreboard untouched in `RL_POLICY/` for history if
desired (move it there manually; not the loop's job).

## Constraints (hard)

1. **Read access:** entire repository.
2. **Write access:**
   - Phase 0 iter 0 only: `calculate_probablities.py` + `train.py`
     (for the two specific fixes above) **and** `event_processing.py`.
   - Phase 1 (iter 1+): only `event_processing.py`. Touching anything
     else invalidates the run.
3. Do not change public function signatures imported elsewhere.
4. Do not introduce new dependencies.
5. Do not use revan or read `.tra` files.
6. Do not silently drop the 7-hit cap without justifying it.
7. Do not run pytest. Do not edit tests.

## Idea backlog

Same as V1: read `EVENT_RECONSTRUCTION_THEORY.md` at the start of
every Phase 1 iteration. Pick the highest-impact filter or
augmentation not already tried (across both V1 and V2 scoreboards).

## Anti-patterns

All V1 anti-patterns still apply. Additionally:

- Do not extend Phase 0 beyond the two named fixes. If a third bug
  is tempting, log it and continue — Phase 0 is *one* commit, not a
  refactor pass.
- Do not skip the post-Phase-0 re-baseline. V1's baseline MCC=0.8663
  is from the buggy pipeline; the V2 baseline will differ and is the
  reference for all V2 guardrails.
- Do not bundle removal + augmentation in one iteration. One verb
  per iter; attribution requires it.
