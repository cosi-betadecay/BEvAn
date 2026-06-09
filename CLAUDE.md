# RL-Style Reconstruction Improvement Loop

You are running an iterative optimization loop on event reconstruction.
Your job is to improve the **eval F1 score** by repeatedly editing
`src/betadecay-analysis/physics/event_processing.py` and re-running the
pipeline. This file is the policy — read it at the start of every iteration.

## Mission

Maximize `eval/f1_score` on the held-out split, subject to the constraints
below. Each iteration: form one hypothesis, edit, evaluate, log result,
keep or revert.

## Reward

**Primary:** `eval/f1_score` from a full pipeline run.

Extract by running:

```bash
python -m betadecay_analysis.run 2>&1 | tee runs/iter_$ITER.log
grep "eval - " runs/iter_$ITER.log | tail -1
```

The line emitted at `modeling/calculate_probablities.py:173` looks like:
`eval - Precision: 0.8123, Recall: 0.7044, FPR: 0.0512, F1 Score: 0.7544`

Parse the F1 value. This is the scalar reward.

**Guardrails (any violation = automatic revert):**
- `eval/recall` must not drop below `baseline_recall - 0.05`
- Fraction of events surviving reconstruction must stay ≥ 50% of baseline
  (compute from log: count of events fed into `predict`)
- All existing tests must pass: `pytest tests/ -x -q`

## Constraints (hard)

1. **Only edit** `src/betadecay-analysis/physics/event_processing.py`.
   No edits to other source files, configs, or tests.
2. Do not change public function signatures imported elsewhere
   (grep first: `grep -rn "from physics.event_processing" src/ tests/`).
3. Do not introduce new dependencies.
4. Do not use revan or read `.tra` files — this branch is `not-using-revan`.
5. Do not silently drop the 7-hit cap (Boggs & Jean) without justifying it
   in the scoreboard entry.
6. No mocking, no test edits to make things pass.

## Per-iteration protocol

Every iteration follows these steps in order:

1. **Read state**
   - Read `RL_SCOREBOARD.md` (full file).
   - Read the current `event_processing.py`.
   - Read the last 5 scoreboard entries to avoid repeating ideas.

2. **Form one hypothesis**
   - State it in ≤2 sentences: what you'll change and why F1 should go up.
   - Reject the idea if it appears (even rephrased) in the last 10 entries.
   - Pull from `Idea Backlog` below if you have no new idea.

3. **Edit** `event_processing.py`. One coherent change per iteration.

4. **Evaluate**
   - `pytest tests/ -x -q` — if fail, revert (`git checkout -- event_processing.py`)
     and log as failed iteration.
   - Run the pipeline and extract F1, recall, survival rate.

5. **Decide**
   - **Keep** if F1 > best-so-far AND guardrails pass: `git add -A && git commit -m "iter N: <hypothesis>, F1=<x>"`.
   - **Revert** otherwise: `git checkout -- src/betadecay-analysis/physics/event_processing.py`.

6. **Log to `RL_SCOREBOARD.md`** (append, never rewrite):
   ```
   ## Iter N — <kept|reverted|failed-tests>
   - Hypothesis: <one line>
   - Diff summary: <one line — what changed, e.g. "lever-arm cut 0.5→0.3">
   - F1: 0.xxxx (Δ vs best: +/-0.xxxx)
   - Recall: 0.xxxx | Survival: xx%
   - Notes: <one line, e.g. "recall dropped — too aggressive">
   ```

## Scoreboard schema

`RL_SCOREBOARD.md` lives at repo root. Header (write once, on iter 0):

```
# RL Scoreboard
Baseline: F1=<x>, Recall=<y>, Survival=<z>%
Best so far: iter <n>, F1=<x>
```

Update "Best so far" line whenever a new best is kept.

## Stop conditions

Halt the loop and report to the user when ANY:
- 50 iterations completed.
- 10 consecutive iterations without a new best.
- 3 consecutive iterations with `pytest` failures (likely something
  structural is wrong; ask the user).
- F1 ≥ 0.95.

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

- Tweaking the same constant repeatedly in tiny steps. Sweep, then move on.
- Bundling multiple changes in one iteration ("I'll change lever-arm AND
  hit cap"). One change per iter — you can't attribute the reward otherwise.
- Editing `physics_factors.py`, `compton_cone_reconstruction.py`, or
  any config to "help" event_processing. If you genuinely need this,
  STOP and tell the user.
- Reporting F1 from training split. Only `eval/f1_score` counts.
- Caching old results — every iteration runs the full pipeline fresh.

## On the first iteration

1. Run the unmodified pipeline once to establish baseline F1, recall, survival.
2. Write the scoreboard header with baseline values.
3. Then begin iter 1 normally.
