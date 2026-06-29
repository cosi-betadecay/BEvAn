---
name: gpu-performance
description: GPU-friendliness, vectorization, and time/space-complexity workflow for this repo — how to make the torch tensor pipeline (datasets → features → density fit → likelihood ratio → eval) run efficiently on GPU WITHOUT changing the interpretable physics-Bayes model or moving the champion F1. Covers profiling first, vectorizing the per-event scalar loops in pipeline/datasets.py, device/dtype/sync hygiene, the host↔device anti-patterns already in the code (per-event torch.cuda.empty_cache, scalar float() conversions), Big-O/space reasoning at each pipeline stage, and the F1-equivalence proof every speedup must pass. Use when working the GPU-friendliness branch, vectorizing a feature or fit, reasoning about complexity/memory, or reviewing a perf diff.
---

# GPU performance for betadecay-analysis

Python 3.10, **torch** tensors throughout. This skill is about making the pipeline
**fast and GPU-friendly without changing what it computes**. It is the engineering
counterpart to `code-quality` (which owns the F1 discipline) and `code-conventions`
(which owns authoring style). Read those two first if you haven't.

## Scope guardrail (read once, then obey)

The model is a fixed, interpretable per-bucket naive-Bayes likelihood ratio. GPU work
here means **vectorizing and relocating the same arithmetic onto the device** — never
re-architecting the model, never trading exactness for speed, never adding a learned
component. If a "perf" idea changes the *numbers*, it is not a perf change; it's a model
change and must go through the normal measure-and-defend path (see `code-quality`). The
non-negotiable acceptance test for every change in this skill is **identical F1** (the
champion; old reference 0.8935) — speed is only allowed to buy us the *same* answer faster.

## The two-number rule

Every speedup must report **both** numbers, in this order:

1. **Correctness:** F1 / AUC unchanged vs the previous run's CSV (`results/<timestamp>.csv`).
   Prefer bit-exact: feature tensors equal to the scalar path on a held sample
   (`torch.allclose(new, old, rtol=0, atol=0)` where the op is genuinely deterministic;
   otherwise `atol=1e-6` and say so).
2. **Speed:** wall-clock before/after on the same `.sim`, same machine, warm cache.

A change that improves (2) but can't prove (1) is **rejected**, not "probably fine."
This mirrors the repo rule: *don't change results while just cleaning up* — re-measure,
don't assume neutral.

## Profile before you touch anything

The dominant cost is **not** in the linear algebra — it's the **per-event Python loop**
in `pipeline/datasets.py` (`compute_event_features`, `compute_event_pool`). Each event is
processed one at a time, each feature is computed as a **Python scalar**, and results are
`list.append`-ed before a single `torch.tensor(...)` at the end. That is `O(N_events)`
Python-level iterations with no batching — the GPU is idle the entire time.

Confirm with a profile, don't guess:

```bash
python -c "import cProfile,pstats; cProfile.run('<extraction call>','prof'); pstats.Stats('prof').sort_stats('cumtime').print_stats(25)"
# torch-level: wrap a hot section in torch.profiler.profile(activities=[CPU,CUDA]) and print key_averages()
```

Hot-path order, by experience with this code:
1. `.sim` reading + PyROOT event objects (I/O bound, hard to vectorize — bounded by MEGAlib).
2. The per-event feature loop (CPU scalar — **this is the real target**).
3. The density fit + likelihood ratio (already array-shaped; vectorizes cleanly, smallest win).

## Concrete anti-patterns already in the tree

These are real, fix them first — they're pure overhead with no correctness stake:

- **`torch.cuda.empty_cache()` inside the per-event loop** (`datasets.py` ~L113 and ~L176).
  Nothing in that loop allocates GPU memory (every feature is a host-side `float`), so this
  is a costly no-op called once per event. Remove it, or hoist to at most once per extraction.
- **`float(delta_E(...))` / `float(arm(...))` / `float(annihilation_angle(...))` per event.**
  Each `float()` / `.item()` forces a **device→host sync**. If a feature ever runs on GPU,
  per-event `float()` serializes the whole pipeline on the PCIe bus. The fix is structural:
  compute features **batched** over many events, keep them as tensors, and convert to host
  only once at the boundary.
- **`list.append` then `torch.tensor(list)` at the end.** Correct, but inherently serial.
  Where event processing can yield arrays, prefer building tensors directly and concatenating.

## What "vectorize" means here

The model already has a vectorized exemplar — copy its shape: `pool_to_features` (`datasets.py`)
relabels and buckets the **entire pool with boolean masks** (`pool["label_score"] < tolerance`,
`pool["bucket"] == b`), no Python loop. That is the target idiom for the feature stage too.

The obstacle is **ragged data**: events have variable hit counts and variable usable subsets,
so features aren't a clean `(N, k)` matrix. Options, in order of preference:
1. **Bucket-batch:** events in the same feature bucket share a feature set — batch within a
   bucket where shapes align.
2. **Pad + mask:** pad hit dimensions to a max and carry a validity mask; reduce with masked
   ops. Watch the space cost (below).
3. **Segment/ragged ops:** `torch.segment_reduce` / index_add over a flat (sum-of-hits) tensor
   with an event-id index — `O(total_hits)` time, `O(total_hits)` space, no padding waste.

## Device, dtype, and sync hygiene

- **One device, decided once.** Resolve `device = "cuda" if torch.cuda.is_available() else "cpu"`
  at the top of a run and thread it through; don't probe `torch.cuda.is_available()` per event.
- **Move once, compute, move once.** The killer is host↔device ping-pong. Read → build a batched
  CPU tensor → `.to(device)` once → all features/fit/likelihood on device → bring back the small
  final scores. Avoid any `.item()`/`float()`/`.cpu()`/`print` of a device tensor in a hot loop.
- **dtype is load-bearing for correctness, not just speed.** Features are `float32`; the label
  score is **`float64`** on purpose (`compute_event_pool`) and the density/log-likelihood sums
  are precision-sensitive — a `float32` reduction over many terms can shift the decision and move
  F1. Keep accumulations in `float64` (or `torch.logsumexp`) unless you've proven `float32` gives
  the identical confusion matrix. **511/1022 keV and the keV tolerances stay keV; angles stay rad.**
- **NaN is a routed signal, not garbage** (see the pipeline contract in CLAUDE.md). Vectorized
  code must preserve `feature_bucket` semantics exactly: `isfinite(anni)→3`, `isfinite(arm)→2`,
  else `1`. Use `torch.isfinite` masks; never `nan_to_num`/`fillna` a feature — that misroutes
  events and poisons a bucket's density. Confirm the per-bucket event **counts** are unchanged
  after vectorizing.

## Reason about complexity at each seam

State the Big-O and the space when you change a stage; put it in a comment at the seam (the
repo prefers an `assert`/short comment over prose):

| Stage | Time | Space | Notes |
|---|---|---|---|
| `.sim` read | `O(N_events · hits)` | `O(hits)` streamed | I/O bound; MEGAlib-limited, leave streamed |
| feature compute (target) | `O(total_hits)` batched | `O(total_hits)` ragged / `O(N·hit_max)` padded | padding wastes space when hit counts are skewed |
| density fit | `O(N_train)` per bucket/term | `O(bins)` | train split only — never the full set |
| likelihood ratio | `O(N · Σ terms)` | `O(N)` | positional term order is fixed; don't reorder |
| eval / thresholds | `O(N log N)` (sort for ROC) | `O(N)` | |

Two traps specific to this repo: **pad-to-max blows up space** when one bucket has a few
high-multiplicity events (prefer segment ops there), and the **likelihood ratio multiplies term
densities positionally** — any vectorized rewrite must keep each bucket's term list in the same
order as the fit, or it silently computes a different product.

## Determinism (a GPU change can move F1 by itself)

CUDA reductions/atomics are not bit-reproducible by default, and a tiny numeric drift can flip a
borderline event and change F1. The repo already seeds `torch.Generator(...).manual_seed(42)` for
splits — keep that. When moving a reduction to GPU, set `torch.use_deterministic_algorithms(True)`
while validating, compare the confusion matrix, and only then accept. If determinism can't be had,
treat any F1 change as **real regression**, not noise.

## Acceptance checklist (paste into the PR)

- [ ] Profiled before/after; named the actual bottleneck (not assumed).
- [ ] Removed/hoisted the per-event `empty_cache`; no `.item()`/`float()` in a hot loop.
- [ ] Per-bucket, per-class **event counts** identical to the scalar path (NaN routing intact).
- [ ] Feature tensors match scalar path (`allclose`, stated tolerance).
- [ ] Accumulations in `float64`/`logsumexp` where decisions are precision-sensitive.
- [ ] **F1 / AUC identical** to the previous `results/` CSV.
- [ ] Wall-clock before/after reported on the same `.sim`/machine.
- [ ] Term order preserved between fit and predict; train-only fit preserved.

## Related skills

- `code-quality` — the F1-regression discipline and the feature→bucket→density→ratio→F1 cascade.
- `code-conventions` — function/dtype/shape-comment authoring style and the assert-at-seams rule.
- `pytorch` — `torch.profiler`, `torch.compile` (inductor vs cudagraphs), device patterns. Use it
  for the torch mechanics; use *this* skill for what's correct/load-bearing in *this* pipeline.
- `compton-physics` / `positron-annihilation-511` — why the units and the NaN routing mean what
  they mean, so a "perf" refactor doesn't quietly break the physics.
