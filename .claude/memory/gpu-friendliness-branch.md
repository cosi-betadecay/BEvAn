---
name: gpu-friendliness-branch
description: "branch-77 GPU work — the real bottleneck, the in-tree anti-patterns, and the gpu-performance skill that governs it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3db6a313-8300-4de8-b26c-da3856211957
---

Branch `77-make-repo-as-gpu-friendly-as-possible` (active as of 2026-06-29) shifts focus
from science (plateaued at the ~0.89 ceiling, see [[betadecay-classifier-state]]) to
engineering: making the torch tensor pipeline run efficiently on GPU **without changing
the model or moving F1**.

The real bottleneck is NOT the linear algebra — it's the **per-event Python loop** in
`pipeline/datasets.py` (`compute_event_features`, `compute_event_pool`): scalar `float(...)`
feature calls + `list.append`, `O(N_events)` serial, GPU idle. Two concrete in-tree
anti-patterns to fix first (pure overhead, no correctness stake): a per-event
`torch.cuda.empty_cache()` inside the loop (~L113, ~L176) and the per-event `float()`/`.item()`
device→host syncs. `pool_to_features` (boolean-mask vectorized) is the target idiom.

Governed by the **gpu-performance** skill (`.claude/skills/gpu-performance/SKILL.md`, written
2026-06-29) — profile-first, vectorize, device/dtype/sync hygiene, Big-O/space per stage, and
the F1-equivalence gate. Pairs with the third-party **pytorch** skill (added via
`npx skillfish add itsmostafa/llm-engineering-skills pytorch`, copied into `.claude/skills/`)
for torch mechanics. Both registered in SKILLS.md.

GUARD: every speedup must prove identical F1/AUC vs the previous `results/` CSV AND report
wall-clock before/after. dtype is load-bearing — keep precision-sensitive sums in float64;
NaN routing (`feature_bucket`) must be reproduced exactly by any vectorized rewrite. See
[[cosi-working-style]] (measure before implementing) and [[hit-multiplicity-bucketing]].
