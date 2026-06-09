# RL-Style Reconstruction Improvement Loop

You are running an iterative optimization loop on event reconstruction.
Your job is to improve the **eval F1 score** by repeatedly editing
`src/betadecay-analysis/physics/event_processing.py` and re-running the
pipeline. This file is the policy — read it at the start of every iteration.

## Mission

Maximize eval **MCC** (computed from the TP/FP/FN/TN that `run.py` prints
on stdout) on the held-out split, subject to the constraints below. Each iteration: form one hypothesis, edit, evaluate, log result,
keep or revert.

**Run autonomously. Do not ask the user questions.** Make the reasonable
call and continue — the scoreboard is the redirect mechanism. If a
genuinely ambiguous decision blocks progress (e.g. a hard stop condition
fires, or you'd otherwise need to edit a file outside `event_processing.py`),
write the situation into the next scoreboard row's Notes column and halt.
The user will read the scoreboard and redirect. Everything else: just keep
iterating.

### What the job actually is

The current reconstruction in `event_processing.py` is **correct**: it
enumerates **all non-empty hit subsets** of each event. That part stays.
The problem is that this catches everything — true β⁺ annihilations *and*
a flood of false positives from background partitions that happen to look
annihilation-like.

**Your one job is to add smart filters that discard bad subsets so
fewer false positives survive into the classifier.** Higher precision
without crushing recall = higher F1.

You ARE expected to **invent and implement new filtering algorithms**
inside `event_processing.py` when the obvious knobs are exhausted —
new geometric cuts, kinematic-consistency tests, energy-window logic,
topology heuristics, multi-stage filters, etc. The Idea Backlog is a
seed, not a ceiling. Write whatever helper functions you need *inside
this one file*.

## Reward

**Primary:** **MCC** (Matthews correlation coefficient) on the eval split,
computed from the raw confusion matrix that `run.py` prints.

Why MCC, not F1: F1 ignores TN, so filters that game the class balance can
look neutral to F1 but are actually worse. MCC uses all four cells, has no
threshold or β to tune, and is harder to game. Range is [-1, 1]; higher is
better.

### How to get the reward

Run the pipeline, capture stdout, and parse the two `eval - ` lines:

```bash
python -m betadecay_analysis.run 2>&1 | tee runs/iter_$ITER.log
grep "^eval - " runs/iter_$ITER.log
```

Those lines come from `modeling/calculate_probablities.py:171-173` and
look like:

```
eval - TP: 1234, FP: 56, FN: 78, TN: 9012
eval - Precision: 0.9565, Recall: 0.9405, FPR: 0.0062, F1 Score: 0.9484
```

Compute MCC from the four counts:

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
```

If any factor in the denominator is zero, treat MCC as 0.

Also parse precision, recall, FPR, F1 from the second line and log them
in the scoreboard as diagnostic columns — they make trade-offs visible
even though MCC is the decision metric.

**Guardrails (any violation = automatic revert):**
- `eval/recall ≥ baseline_recall - 0.05` (don't crush signal to chase MCC).
- `run.py` must complete without errors.

**No pytest. No test edits.** The only evaluation is running `run.py` and
parsing its stdout.

## Constraints (hard)

1. **Read access: entire repository.** You are encouraged to read any
   file in the repo — `physics_factors.py`, `compton_cone_reconstruction.py`,
   `modeling/`, `pipeline/`, configs, docs, tests — to understand how
   subsets flow downstream and what features your filters should preserve.
   Reading is unlimited; **editing is not**.
2. **Write access: exactly one file** —
   `src/betadecay-analysis/physics/event_processing.py`. No edits anywhere
   else, ever.
3. Do not change public function signatures imported elsewhere
   (grep first: `grep -rn "from physics.event_processing" src/ tests/`).
4. Do not introduce new dependencies.
5. Do not use revan or read `.tra` files — this branch lineage is `not-using-revan`.
6. Do not silently drop the 7-hit cap (Boggs & Jean) without justifying it
   in the scoreboard entry.
7. Do not run pytest. Do not edit tests.

## Per-iteration protocol

Every iteration follows these steps in order:

1. **Read state**
   - Read `SCOREBOARD.md` (full file). The table is your memory across
     iterations — scan the "What changed" and "Notes / next idea"
     columns to avoid repeats and pick up promising threads.
   - Read the current `event_processing.py`.

2. **Form one hypothesis**
   - State it in ≤2 sentences: what you'll change and why F1 should go up.
   - Reject the idea if it appears (even rephrased) in the last 10 entries.
   - Pull from `Idea Backlog` below if you have no new idea.

3. **Edit** `event_processing.py`. One coherent change per iteration.

4. **Evaluate**
   - Run `run.py`. Parse the two `eval - ` lines from stdout to get
     TP/FP/FN/TN, then compute MCC. Also record precision, recall, FPR, F1.
   - If `run.py` errors out, revert (`git checkout -- event_processing.py`)
     and log the row as `reverted` with the error in Notes.
   - Do NOT run pytest.

5. **Decide**
   - **Keep** if MCC > best-so-far AND guardrails pass: `git add -A && git commit -m "iter N: <hypothesis>, MCC=<x>"`.
   - **Revert** otherwise: `git checkout -- src/betadecay-analysis/physics/event_processing.py`.

6. **Log to `SCOREBOARD.md`** — append one row to the table (never
   rewrite or reorder existing rows).

## Scoreboard schema

`SCOREBOARD.md` lives at repo root. It is a single markdown table plus
a one-line "Best so far" pointer above it. Write the header once on
iter 0; append one row per iteration thereafter.

```
# Scoreboard

Best so far: iter <n>, MCC=<x>

| Run ID | Timestamp (UTC) | Status | Hypothesis | What changed | MCC | Δ vs best | F1 | Precision | Recall | FPR | TP / FP / FN / TN | Notes / next idea |
|--------|-----------------|--------|------------|--------------|-----|-----------|----|-----------|--------|-----|--------------------|-------------------|
| 0 | 2026-06-09T12:34:56Z | baseline | — | unmodified repo | 0.8412 | — | 0.7544 | 0.8123 | 0.7044 | 0.0512 | 1234 / 56 / 78 / 9012 | starting point |
| 1 | … | kept | lever-arm too lax, tighten to 0.3 cm | `min_lever_arm: 0.5→0.3` | 0.8487 | +0.0075 | 0.7611 | 0.83 | 0.70 | 0.04 | … | recall held — try 0.2 next |
| 2 | … | reverted | aggressive energy-window | added ±20 keV window around 511 | 0.83 | -0.01 | … | … | … | … | … | dropped MCC, recall tanked |
| 3 | … | reverted | new geometric filter | added back-to-back precut | — | — | — | — | — | — | — | run.py crashed: NaN in cos_theta; retry with clamp |
```

Rules:
- **Run ID** = iteration number, monotonically increasing. Iter 0 is the
  untouched baseline.
- **Status** ∈ {`baseline`, `kept`, `reverted`}. Crashes, guardrail
  violations, and no-improvement runs all collapse to `reverted` — note
  the reason in the Notes column.
- **What changed** is the actual diff in one line (constant change,
  filter added, function introduced). Be specific enough that a future
  iteration can tell whether an idea was already tried.
- **Δ vs best** is signed MCC delta vs the best-so-far row at the time
  of this iteration (not vs the previous row).
- **Notes / next idea** is where you record what you learned and, if
  useful, the next thing to try. This column is the planning log —
  future iterations read it to avoid repeats and find threads to pull.
- Update the **"Best so far"** line whenever a kept row beats it.
- Never edit prior rows. If a number was wrong, add a new row noting it.

## Stop conditions

Halt the loop and report to the user when ANY:
- 50 iterations completed.
- 10 consecutive iterations without a new best.
- 3 consecutive iterations where `run.py` errors out (likely something
  structural is wrong — log it in the scoreboard and halt, do not ask).
- MCC ≥ 0.95.

## Idea backlog (seed — extend, don't exhaust blindly)

Reconstruction levers actually inside `event_processing.py`:

- **Lever-arm cut**: sweep 0.1, 0.3, 0.5 (current), 0.75, 1.0 cm.
- **Hit cap**: try 5, 6, 7 (current), 8.
- **Subset enumeration strategy**: instead of *all* non-empty subsets,
  enumerate only subsets that partition hits into two groups (two-photon
  prior) and score each partition.
- **Per-subset ordering**: within a subset, sort hits by distance from
  origin / by energy / by a Compton-kinematic χ², instead of leaving
  order arbitrary.
- **Energy-window pre-filter**: drop subsets whose ΣE is > N keV from
  511 or 1022 keV.
- **Escape rejection**: drop subsets where the last hit's energy
  exceeds Compton-edge for the running residual.
- **Minimum hit count**: require ≥3 hits (current `annihilation_angle`
  silently NaNs sub-3).
- **Two-photon back-to-back pre-filter**: pre-cut subset pairs whose
  best back-to-back cos > -0.9.

Mark each in the scoreboard as `tried` when used, even if reverted.

## Anti-patterns (do not do)

- **NEVER EDIT ANY FILE OTHER THAN `src/betadecay-analysis/physics/event_processing.py`.**
  Not `physics_factors.py`. Not `compton_cone_reconstruction.py`. Not
  configs, not tests, not `run.py`, not `pyproject.toml`, not docs, not
  CLAUDE.md, not the scoreboard's structure. **Zero exceptions.** Reading
  any file in the repo is fine and encouraged — editing is forbidden.
  If you genuinely believe a change outside this file is required, STOP
  the loop and report to the user. Violating this rule invalidates the
  entire run.
- Removing or weakening the subset enumeration itself. The enumeration is
  correct; your job is to **filter** its output, not replace it.
- Tweaking the same constant repeatedly in tiny steps. Sweep, then move on.
- Bundling multiple changes in one iteration ("I'll change lever-arm AND
  hit cap"). One change per iter — you can't attribute the reward otherwise.
- Reporting metrics from the training split. Only the `eval - ` lines count.
- Caching old results — every iteration runs the full pipeline fresh.

## On the first iteration (iter 0 = baseline)

**Do not touch any code on the first run.** Iter 0 is a pure baseline:

1. Run `run.py` on the unmodified repo.
2. Parse TP/FP/FN/TN from stdout, compute MCC, also record F1,
   precision, recall, FPR.
3. Append iter 0 to `SCOREBOARD.md` as the baseline row (no diff, no
   hypothesis — just "baseline, no changes").
4. Use these numbers as `baseline_mcc`, `baseline_recall` for guardrail
   checks and as the initial "Best so far" row.
5. Only then begin iter 1 with your first hypothesis.
