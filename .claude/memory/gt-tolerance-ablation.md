---
name: gt-tolerance-ablation
description: "branch 75 gt_tolerance label-window sweep — robust (all metrics ≥0.8, AUC flat ~0.99), n_std=3 reproduces champion; tighter-window F1 gains are label artifacts, NOT new champions"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec6092d0-1cc5-4b39-b8ce-d5f10274ee6e
---

The `gt_tolerance` ablation (ablations/gt_tolerance.py) sweeps the ground-truth 511 keV
label window `n_std` ∈ 1..10 and refits/evaluates per value. Built + validated on VM
2026-06-27.

**Design:** only the LABEL depends on n_std (features + hit-multiplicity bucket don't), so
the .sim is streamed ONCE per dataset into a label-free pool (features + bucket + per-event
`bdecay_label_score` = min|ANNI-sum−511|), then relabeled in-memory per window
(`pool_to_features` + `split_features`). 1 stream, 10 fits — not 10 streams. E_TOLERANCE
(feature deadzone) held fixed at 3σ → isolates the label, model unchanged. Verified
byte-identical to fresh per-window extraction (offline) AND `n_std=3` column reproduces the
champion exactly on real data.

**Result = clean robustness:** all four metrics (F1/AUC/precision/recall) stay ≥0.8 across
ALL 10 windows, AUC flat ~0.99 (tight bands = geometries agree), deployed 3σ sits mid-plateau
(not a peak). F1/precision/recall move as physics predicts (tighter window → precision down
then up, recall up then down, mechanical relabel effect) but stay in-band. Defeats the "your
result is an artifact of an arbitrary cut" attack. 3σ stays the deployed label (3σ of HPGe
resolution, physically motivated) — see [[betadecay-classifier-state]], [[energy-specificity-vs-generic-geometry]].

**GUARD — do NOT chase tighter-window F1.** SPILike@n_std=2 scored 0.908 F1 (vs SPILike@3σ
~0.8675, vs the 0.8935 Max champion). This is NOT a new champion: tightening the window
relabels ambiguous near-511 events out of signal → cleaner/easier separation → higher F1. The
gain is a property of the LABEL, not the model; adopting the max-F1 n_std would tune the ground
truth to inflate the metric and torpedo the ablation's own credibility. SPILike is just the
most label-sensitive geometry (coded mask + BGO, false-511 contamination) — a physics
observation, not a model win. (User's old simpler 1σ-posterior pipeline hit 0.905 F1 via heavy
F1-tweaking/overfit; off-label trophy, same caveat.)
