# RL Policy V3 — Two-photon Joint Constraint Loop

V1 and V2 ran on `physics/event_processing.py` and converged at
**MCC = 0.8785** (V2 iter 13). The convergence is structural, not
hyperparameter-limited: of the three classifier features
(`delta_E`, `annihilation_angle`, `arm`), only `arm` is order-sensitive,
so it is the only one `event_processing.py` can influence. `delta_E`
and `annihilation_angle` are globally optimal over the enumerated
subset pool from `event_processing.py`'s perspective — untouchable
from that file.

The residual ~117 FP are background events that look 511-like in all
three features *independently*. The classifier's per-feature `min`
aggregation never asks whether the *same* hit-pair satisfies both the
back-to-back angle constraint **and** the 511 keV energy constraint —
random Compton coincidences can pass each independently and slip
through. The theory document (`EVENT_RECONSTRUCTION_THEORY.md`, §7)
flags this exact gap: real annihilation produces two ~511 keV photons
back-to-back, summing to ~1022 keV. Coupling angle and energy at the
*same pair* is the next discriminating lever.

V3 moves the loop downstream — to `physics_factors.py` — and focuses
on **`annihilation_angle`** specifically.

**Trigger:** when the user says `Execute RL_V3.md` (or "start V3",
"run the V3 loop", "go"), begin at Phase 0. Until then, this is a
normal doc.

---

## Mission

Maximize eval MCC by editing the `annihilation_angle` function (and
its immediate call site to thread energies through) so that the
returned per-event angle reflects a **two-photon joint constraint**:
the chosen back-to-back pair must also be energy-consistent with a
1022 keV annihilation total.

Guardrail: recall ≥ V3 baseline − 0.05. `run.py` must complete end-to-end.

## Baseline

The V3 baseline is **the current `rl-cli` HEAD** (V2 iter 13,
MCC = 0.8785, recall = 0.9536, FP = 117). All V2 changes to
`event_processing.py` stay in place — V3 builds *on top of* the
ordering-filtered subset pool, it does not revert it.

## Phase 0 — thread energies into `annihilation_angle` (iter 0 only)

The current `annihilation_angle(positions, n_hits, sizes)` signature
does not receive energies, so no joint constraint is possible without
plumbing. Phase 0 is the one-shot plumbing pass.

You are permitted — **only on iter 0 of V3, only for this specific
edit** — to modify:

- `src/betadecay-analysis/physics/physics_factors.py`
- `src/betadecay-analysis/dataset/datasets.py` (call site only)

The Phase 0 change:

1. Extend `annihilation_angle` to accept an optional `energies`
   argument with shape matching `positions` (per-hit energies).
2. Update the single call site in `dataset/datasets.py` (around
   line 75) to pass `energies` through.
3. Phase 0 must be **behavior-preserving**: when `energies` is
   provided, the function must still return the same cosine value
   as before (ignore the new argument for now). Verify by re-running
   `run.py` and confirming MCC is byte-identical to the V3 baseline
   (0.8785). If it isn't, the plumbing changed behavior — stop and
   debug before Phase 1.

Phase 0 deliverable: one commit `phase0: thread energies into
annihilation_angle (no-op)`, pushed to `rl-cli`. Lock the writable
surface back down. From iter 1 onward, the only editable file is
`src/betadecay-analysis/physics/physics_factors.py`, and within it
only the `annihilation_angle` function body.

If the plumbing requires changing a public signature elsewhere or
breaking an existing caller, stop and report — do not improvise.

## Phase 1 — iteration loop

Same per-iter protocol as V1/V2: read state, one hypothesis, edit,
evaluate, decide, log. Same commit rule: push only on a new best.

### Action space — joint angle/energy constraint inside `annihilation_angle`

The function returns a cosine (closest-to-back-to-back over candidate
pairs/triples). V3 changes *which candidates are eligible* for that
minimum, using the energies now available. The return type, scale
([-1, 1]), and "lower-is-better" convention stay the same — downstream
density-matrix binning is calibrated to those, and binning lives
outside the writable surface.

Seed hypotheses, ranked by theory leverage:

1. **Hard 1022 keV pair gate.** Only pairs whose summed energy is
   within ±σ_E of 1022 keV are eligible for the cosine min. Try
   σ_E ∈ {30, 50, 80, 120} keV. Fallback if no pair passes: return
   NaN (event drops out — same convention the existing function uses
   for unreachable subsets).
2. **Soft Gaussian weight.** Eligible cosine of a pair is
   `cos + λ · (1 − exp(−(E_i + E_j − 1022)² / (2σ_E²)))` — penalizes
   off-energy pairs continuously instead of hard-gating. Sweep λ.
3. **Per-photon 511 prior.** Require each individual hit (or each
   half of the pair, summed within a subset) to be within ±σ of 511,
   not just the joint sum. Tighter than (1).
4. **Triple/chain joint constraint.** For size ≥ 3, score the pair
   formed by hit-i + Σ(rest) and hit-j + Σ(rest) — i.e. treat the
   two reconstructed Compton chains as the two photons and check
   their summed energy against 1022. Engages the multi-hit subsets
   that pair-only hypotheses ignore.
5. **Asymmetry exploit.** If both photons must be ~511, reject pairs
   where one hit's energy > 511 + δ (impossible for a single photon
   pre-scatter, theory §2). Targets specific FP topologies.

The pattern that worked in V2 — **filter the candidates, don't change
the aggregation shape** — applies here. Don't replace the cosine min
with a different scalar; restrict the set the min runs over.

### Anti-repeat

Do not re-try anything from V1's or V2's scoreboards (those are about
subset enumeration in `event_processing.py`, structurally distinct,
but the *removal-is-zero-sum* lesson transfers: if a hypothesis only
prunes candidates without adding new ones, expect ~1:1 TP:FP unless
the prune is genuinely asymmetric on the signal/bg energy distribution
— which is exactly what the 1022 keV constraint is hypothesized to be).

### Stop conditions

- 50 Phase 1 iterations completed.
- 10 consecutive Phase 1 iterations without a new best.
- 3 consecutive iterations where `run.py` errors out (re-run once
  before counting an error).
- MCC ≥ 0.95.
- Recall floor breach (recall < 0.9036) for two consecutive iters.

## Scoreboard

Use a fresh `SCOREBOARD_V3.md` at repo root. Same schema as V1/V2.
Iter 0 row is the **post-Phase-0 baseline** (should match V2 iter 13
exactly, since Phase 0 is a no-op).

## Constraints (hard)

1. **Read access:** entire repository.
2. **Write access:**
   - Phase 0 iter 0 only: `physics_factors.py` (the `annihilation_angle`
     function and its imports/signature) + `dataset/datasets.py`
     (the single call site).
   - Phase 1 (iter 1+): only the *body* of `annihilation_angle` in
     `physics_factors.py`. Module-level helpers added to support it
     are allowed inside the same file, but no other file may be
     touched. Do not modify `arm`, `delta_E`, or any other function
     in `physics_factors.py`.
3. Do not change public function signatures consumed elsewhere
   (beyond the Phase 0 `energies` addition, which must be a
   keyword arg with a default of `None` for backward compat).
4. Do not change the return type, dimensionality, scale, or
   "lower-is-better" convention of `annihilation_angle`. Downstream
   binning (`conftest.py`, `matrix_calculations.py`) assumes a
   cosine-like scalar in [-1, 1] with linear spacing — none of those
   files are editable.
5. Do not introduce new dependencies.
6. Do not use revan or read `.tra` files.
7. Do not run pytest. Do not edit tests.
8. `event_processing.py` is **frozen** at V2 iter 13. Do not touch it.

## Idea backlog

Read `EVENT_RECONSTRUCTION_THEORY.md` §2, §7, and §3 at the start of
every Phase 1 iteration. §7 (two-photon coincidence) is the primary
theoretical anchor for V3. Pick the highest-impact hypothesis not
already tried (V3 scoreboard).

## Anti-patterns

All V1/V2 anti-patterns apply. Additionally:

- Do not extend Phase 0 beyond the named plumbing edit. If a behavior
  change sneaks in, the byte-identical-baseline check will catch it.
- Do not replace the cosine return with a different scalar (energy
  residual, χ², etc.). The downstream density matrices are calibrated
  for a cosine in [-1, 1]; changing the scale silently breaks binning
  and the "improvement" will be a binning artifact, not a real signal.
- Do not bundle two hypotheses (e.g. hard gate + soft penalty) in one
  iter. One verb per iter; attribution requires it.
- Do not migrate work outside `annihilation_angle` mid-loop. If a
  hypothesis requires a coherence aggregator or a new feature, log
  it as out-of-scope and propose V4 — do not silently expand V3's
  surface.
