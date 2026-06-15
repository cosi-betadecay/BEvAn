# RL Policy V6 — Eyes first, then the legacy terms get the V5 treatment

V5 succeeded (+0.0065, the first bar beaten since V3) and its entire gain
came from ONE mechanism on the model side: the LOR feature contributes only
its conditional P(lor|ΔE) information, with Laplace-smoothed, finely-rowed
densities. Every feature-side hypothesis lost; the iter-9 feature state has
now survived two full loops unimproved. Meanwhile the loop ran blind twice
over: (1) we never knew WHICH 107 eval FP remain, and (2) the harness
pre-flight gates on marginal TV, which has inverted against eval MCC six
documented times across V4+V5 (V4 iters 1/7/8/10, V5 iters 4/16 — iter 16
produced the best-ever truth axis, 6.8°, and still LOST MCC).

V6 therefore does two things, in order:

1. **Phase 0 builds the eyes** (measurement only, no classifier change):
   per-event FP characterization + a conditional-on-ΔE separability metric
   in the pre-flight. This is V5's top-ranked debt, and it is cheap.
2. **Phase 1 applies the proven V5 recipe to the two legacy terms** (angle,
   arm) — the only model-side surface V5 was forbidden to touch. The same
   two arguments transfer verbatim: their joints carry redundant ΔE votes
   that subsidize the 511-like FP class, and their unsmoothed 25×25
   densities have thin-row noise that α=0.5 demonstrably converts into
   generalization. V4 iter 5's single-ΔE failure (−0.0250) is NOT a
   counterexample: that test removed the second ΔE vote with sparse,
   UNSMOOTHED conditionals and no row tuning — exactly the failure mode
   iters 9/11/12 later fixed for the lor term.

**Trigger:** when the user says `Execute RL_V6.md` (or "start V6", "run the
V6 loop", "go"), begin at Phase 0. Until then, this is a normal doc.

---

## Mission

Maximize eval MCC by (a) instrumenting the residual-FP population and
(b) restructuring how the density model consumes the THREE existing feature
pairs (angle, arm, lor) — term form, smoothing, per-pair binning. No new
features. **Success bar: beat MCC 0.8857.**

Guardrail: recall ≥ 0.9036. `run.py` must complete end-to-end.

## Baseline

The V5 champion at current `rl-cli` HEAD (commit `937ca43`): **MCC 0.8857,
recall 0.9536, F1 0.8949, 575 / 107 / 28 / 5885.** Scoreboard "best" starts
at 0.8857; commit/push only on beating it.

Model state, for reference: R = P(ΔE,angle)·P(ΔE,arm) joint ratios (ΔE = 2
votes, `double_count_deltaE: true`) × P(lor|ΔE) conditional ratio
(α=0.5-smoothed, 40×25, floors 1e-3/1e-3). Angle/arm joints: unsmoothed,
25×25, floors 1e-3 (x), linear-y (angle) / 1e-3 log-y (arm).

## Phase 0 — instrumentation (iters 0a/0b, measurement only)

No classifier behavior may change in Phase 0. Two deliverables:

1. **`scripts/fp_census.py`** — replicate the run.py split exactly (same
   seed-42 permutation, same 80/20), classify with the champion model, and
   dump per-event rows for every eval FP and FN: event index, n_hits, ΔE,
   angle, arm, lor, and each term's log-ratio contribution. Summarize: which
   single term flips each FP (would it survive with that term removed?),
   n_hits/ΔE-band histograms, zero-bin membership. Output to
   `outputs/fp_census_v6.json` + printed table.
2. **`scripts/lor_preflight_metrics.py`** — promote the V5 /tmp metrics
   script into the repo and ADD the count-weighted conditional-on-ΔE TV
   (15 ΔE-quantile bins, ≥30/class per bin — the `compare_angle_features.py`
   protocol) for all three features. The pre-flight gate stays pytest 3/3;
   cond-TV joins TV/axis as a LOGGED diagnostic. After 3 iters, evaluate
   whether cond-TV tracks MCC; if it inverts like marginal TV, say so in
   the scoreboard and stop trusting it.

Log iter 0 as the baseline re-measurement (must reproduce 575/107/28/5885
byte-for-byte; if it does not, STOP and report). The FP census findings go
in the iter-0 notes and steer Phase 1 ordering.

## Phase 1 — iteration loop

Per-iter protocol as V1–V5. Pre-flight only when `annihilation_lor.py`
changes (not expected); model-side iters carry harness values. One verb per
iter; MCC decides keep/revert.

### Action space, ranked (re-rank after the FP census)

1. **Arm term → conditional P(arm|ΔE) with the V5 recipe** (α=0.5, then
   row-count probe 25→40 on the ΔE axis of the arm pair). ΔE drops to 2
   votes total only if BOTH legacy joints stay; going arm-conditional makes
   ΔE votes = angle-joint only + ΔE appears once — i.e. 2 votes counting
   the angle joint's implicit one. The lor precedent says the conditional
   keeps the term's information while starving the 511-like FP subsidy.
2. **Angle term → conditional P(angle|ΔE)**, same recipe, same argument.
   NOTE: if both 1 and 2 are applied, ΔE has ONE explicit vote left —
   re-add it as the smoothed 1D marginal ratio (the `build_density_matrix_1d`
   path already exists) so ΔE evidence is preserved exactly once. That
   end-state is the clean factorization V4 iter 5 attempted, now with
   smoothing and tuned rows. Build it stepwise (1, then 2, then the
   marginal re-add) — three iters, each individually revertable.
3. **Smoothing-only probes on the legacy joints** (α=0.5 on angle/arm
   joints, no term-form change) — the minimal-risk version of 1/2 if the
   conditionals lose. Cheap, and V5 iter 9 says thin-row noise is real.
4. **Per-pair binning descent for whichever legacy terms changed form**
   (ΔE rows 25→40→elbow; feature axis stays 25 — both V5 probes of finer
   feature axes lost).
5. **FP-census-driven hypotheses** — whatever the census says the 107 FP
   have in common. This slot is deliberately unspecified until iter 0
   reports; if the census contradicts the ranking above, the census wins.

### Anti-repeat (V4+V5 axes closed — do not re-open)

- Feature-side `annihilation_lor.py` changes: joint (i,k) selection,
  degenerate split, lever-arm penalty, exhaustive internal ordering,
  χ²/dof, σ sweeps, NaN sentinels — ALL refuted. The iter-9 feature state
  is frozen this loop.
- lor term: marginal form, joint form, α ∈ {0.25, 1.0}, x-rows 60, y-bins
  {12, 40}, y-floor {1e-4, 1e-2} — closed at the V5 optimum.
- "Purify the zero bin" in any costume (three failures: V5 iters 1/6/7).
- Re-pricing NaN populations at the density layer (V4 iter 4).
- n>12 multiplicity tail: EMPTY in this file. Unfalsifiable, skip.

### Stop conditions

- 50 Phase 1 iterations; 10 consecutive without a new best; 3 consecutive
  `run.py` errors (re-run once first); MCC ≥ 0.95; recall < 0.9036 twice
  consecutively.

**Failure path:** revert the pipeline to the V5 champion (`937ca43` state),
KEEP the Phase-0 tooling (it is measurement, commit it regardless once
iter 0 reproduces the baseline), write the halt summary. V7 candidates to
log, in order: (a) whatever the FP census surfaced that V6 could not
exploit, (b) cross-feature 2D terms not involving ΔE (e.g. P(lor, angle))
— new modeling surface, (c) source-volume prior on the LOR (still last:
the dangerous bg is no-ANNI coincidences).

## Scoreboard

Fresh `SCOREBOARD_V6.md` at repo root, V5 schema + one new column:
**cond-TV** (the three-feature conditional separability triple from the
new pre-flight script). Iter 0 = baseline re-measurement + FP census
summary.

## Constraints (hard)

1. **Read access:** entire repository.
2. **Write access:** Phase 0: `scripts/fp_census.py`,
   `scripts/lor_preflight_metrics.py`, `outputs/` artifacts. Phase 1:
   `pipeline/train.py`, `pipeline/eval.py`,
   `modeling/calculate_probablities.py`, `configs/config.yaml` — model-side
   lines only. Nothing else.
3. **All four features are frozen as features:** `annihilation_lor.py`
   (iter-9 state), `physics_factors.py`, `event_processing.py`,
   `dataset/datasets.py`. V6 changes how densities consume them, never
   their values.
4. Truth (IA records) never feeds the runtime path; the FP census may use
   labels for GROUPING but its outputs must not parameterize the model.
5. No new dependencies. pytest: only
   `tests/tests_specialized/test_annihilation_lor.py`. Do not edit tests.
6. One verb per iter; commit/push only on a new best (> 0.8857), except
   the Phase-0 tooling commit (allowed once iter 0 reproduces baseline).

## Anti-patterns

All V1–V5 anti-patterns apply. Additionally:

- Do not trust marginal TV for ANYTHING but the >0.05 sanity gate (six
  inversions). Treat cond-TV as on probation until it earns three
  MCC-concordant calls.
- Do not change two terms' form in one iter — the 1→2→marginal-re-add
  ladder in action #2 exists so each rung is individually attributable.
- Do not let the FP census mutate into a fitted discriminator (constraint
  4): it explains, it does not classify.
- If both legacy conditionals lose even with the V5 recipe, accept that
  the double-counted joints are load-bearing for the legacy pair geometry
  and pivot to action #5 rather than grinding binning variants.
