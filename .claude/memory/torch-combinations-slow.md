---
name: torch-combinations-slow
description: "GUARDRAIL — don't replace itertools.combinations with torch.combinations; it's ~8000x slower here"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2b355095-19ee-4e16-9b11-ac2be6cb8a06
---

On branch-77 (GPU-friendliness), `torch.combinations` was tried as a vectorized
replacement for `itertools.combinations` in `physics/event_processing.py`'s subset
enumeration. **It was ~8000x SLOWER** (benchmarked n=14: itertools generation ~2 ms
vs torch.combinations + `.tolist()` ~17,000 ms). It materializes the whole tensor
through an inefficient path and the `.tolist()` round-trip back to Python compounds
it. Reverted to `itertools.combinations`.

**Why:** the per-event subset enumeration is inherently serial Python that needs
CPU scalar access (orderings/`ckd_residual`); a tensor primitive that builds-then-
converts is strictly more work than a lazy C generator.

**How to apply:** keep `itertools.combinations` for generation. Do NOT re-propose
`torch.combinations`. The one salvageable win from that attempt: building the padded
`idx` in a single `torch.tensor(padded)` instead of a per-combo scatter is ~7x
faster and bit-identical — that was kept.

The genuine cost split (n=14, per high-mult event): orderings()+ckd_residual ~85%,
padding ~14%, generation ~1%. The real speedup target ("Lever 2") is vectorizing
the CKD ordering/residual on GPU — but only pays off on the high-multiplicity tail
and must be profiled end-to-end first (I/O and GPU reductions may dominate total
wall-clock). See [[gpu-friendliness-branch]] and [[cosi-working-style]] (measure
before implementing).
