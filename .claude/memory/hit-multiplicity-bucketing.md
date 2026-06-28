---
name: hit-multiplicity-bucketing
description: "Classifier architecture (2026-06-21): events are bucketed by feature availability (n=1 ΔE only / n=2 ΔE+ARM / n≥3 ΔE+ARM+anni) with a separate density model + class prior per bucket, instead of one pooled model. Targets low-multiplicity FP."
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a4769c0-c4dd-4b11-b3f8-5f862b7c1e69
---

Implemented per the user's idea to fix FP: stop pooling all events into one
density model. Right after event processing, route each event to a hit-multiplicity
bucket and give each bucket its own matrices + class prior. Motivation: a lone 511
keV background hit had its (NaN) geometry features floored and was judged in a
pooled model -> FP. Per-bucket priors (background is median 1 hit; ~75% of bg is
<3 hits) prior-crush low-n events toward background.

**Buckets by FEATURE AVAILABILITY** (not raw GetNHTs — robust to anni's energy
gate NaN-ing a >=3-hit event): bucket 1 = ΔE only; 2 = ΔE+ARM; 3 = ΔE+ARM+anni
(anni = the vertex-anchored back-to-back score). `_feature_bucket` in datasets.py.

**Files (all rewritten this session):**
- `dataset/datasets.py`: `compute_event_features` returns nested dict
  `data[class][bucket][feature]` of tensors; `split_dataset(data)` splits 80/20
  within each (class,bucket). Module consts BUCKETS=(1,2,3), _FEATURES, _CLASSES.
  Bucket+label assigned together per event => can't desync (the alignment worry).
- `pipeline/train.py`: `Trainer.fit` builds one model per bucket
  (`self.models[b] = {n_beta, n_bg, terms}`); generic term specs (1d/2d). ΔE
  double-counted in bucket 3 (two 2D joints), once elsewhere.
- `pipeline/eval.py`: loops buckets, accumulates confusion, logs per-bucket
  (`eval/n1` etc.) + summed total.
- `modeling/calculate_probablities.py`: added `confusion_counts` (no logging) +
  `log_confusion`; kept `R`/`predict` for back-compat.
- `run.py`: simplified to `data=...; train,eval=split_dataset(data)`.

**REVERTED 2026-06-22:** the 4-bucket + φ + >=3-merge work below was reverted to
the 3-bucket far-field-ARM peak (commit 86bb140, eval F1 0.8935) — see
the reverted near-field ARM rework. Shipped state is again: 1=ΔE, 2=ΔE+ARM (far-field),
3=ΔE+ARM+anni, _N_BINS={1:25,2:8,3:35}. The update note below is HISTORICAL.

**UPDATE 2026-06-21 (now 4 buckets + 2-hit Compton φ) [REVERTED — historical]:** after the ARM rework
(the reverted near-field ARM rework) ARM needs >=3 hits, which dumped 2-hit events into
the ΔE-only bucket and cost ~0.005 F1 (they lost their dedicated prior). Fix: gave
2-hit its own bucket with a real Compton feature instead of ΔE-only. New `compton_phi`
in physics_factors.py = 2-hit Compton scatter angle from the energy partition
(best valid ordering by Klein-Nishina, MEGAlib's rule; φ=arccos(1-511(1/E_out-1/ΣE))).
Energy-only, blind. AUC 0.60 on 2-hit (KN itself was anti-discriminating 0.27, dropped).
Derived from MEGAlib Revan source at /Users/aryaraeesi/Documents/COSI/megalib
(MERCSRChiSquare.cxx, MComptonEvent.cxx IsKinematicsOK / KN). Buckets (CORRECTED to 3 — do NOT split >=3 into anni/no-anni strata, that
fragments the prior and cost F1): 1=ΔE(1-hit), 2=ΔE+φ(2-hit),
**3=>=3-hit: ONE stratum where BOTH P(ΔE,ARM) and P(ΔE,anni) multiply into R**.
anni is a PER-EVENT OPTIONAL FACTOR: when anni is NaN (no back-to-back), that
factor is dropped (ratio->1) for that event — the event stays in bucket 3 and is
NOT excluded. Implemented in eval.py: global valid = isfinite(delta_E) only;
each term neutralized (pb=pg=1) where its feature is NaN. Routing: arm finite=>3,
elif phi finite=>2, else 1 (anni not used for routing). _N_BINS={1:25,2:10,3:35}
(tunable). Validated torch==numpy + synthetic (NaN-anni events kept in bucket 3);
NOT yet run on VM.

**RESULT (2026-06-21, pre-4-bucket):** reached eval F1 **0.8935** (train 0.901) with per-bucket
bins `_N_BINS = {1: 25, 2: 8, 3: 35}`, marginal smoothing 0.0, joint smoothing 0.5.
This MATCHES the old LOR champion (eval F1 0.8949, the deleted LOR feature)
but WITHOUT LOR — i.e. champion-level metric on a physically-defensible
architecture. Key lever: match bin resolution to each bucket's sample size —
coarse where signal is sparse (n=2→8), fine where rich (n=3→35). Finer n=3 bins
genuinely improved separation (a frontier PUSH, not just a precision/recall
slide). Train/eval gap ~0.7% = healthy, not overfit; watch that gap if pushing
n=3 bins higher. FP localized: ~all remaining FP are n=3 (multi-hit real-511
no-ANNI background = the documented per-event physics ceiling); n=1 contributes
0 FP. Tests still broken by the refactor (old tuple API + deleted LOR test).

**ΔE double-count is LOAD-BEARING — MEASURED A/B 2026-06-22 (do NOT "fix"):**
bucket 3 multiplies P(ΔE,arm)·P(ΔE,anni) so the ΔE likelihood ratio is counted
twice. The textbook de-double-count P(ΔE)·P(arm|ΔE)·P(anni|ΔE) (via the unused
`conditional_from_joint` helper, gated on `BETADECAY_FACTOR_DELTAE`) was run on the
VM: it cut n3 FP 97→83 but ADDED 34 FN (n3 recall 0.965→0.904), dropping eval F1
**0.8935→0.8735**. Reverted. Reason: n3's ΔE≈511 events are mostly real signal, so
up-weighting ΔE helps — the "bug" is correct for this data. ALL session code was
reverted to 83b5d58 (the 0.8935 champion) on user request: the factored model AND
the threshold-sweep diagnostic are GONE from the tree — only these learnings persist.
(Sweep was a τ-multiplier on the MAP threshold `R ≥ τ·n_bg/n_beta`, env
`BETADECAY_THRESHOLD_SWEEP`; factored model env `BETADECAY_FACTOR_DELTAE`; both
trivially rebuildable.)

**n3 FP are degenerate (threshold sweep, opt-in `BETADECAY_THRESHOLD_SWEEP=1`,
read-only):** eval n3 F1 PEAKS at the shipped MAP rule (τ=1, 0.9014); every τ>1
lowers it (~2.3 TP lost per FP removed). Confirms the per-event ceiling — no
operating-point move on n3 buys F1. τ=2 (FP 97→87, F1 −0.01) is only useful as a
deliberate high-purity sample, not an F1 win.

**Overall F1 is POOLED, not per-bucket — n3 is NOT the lever for the total.**
n3 is already the best bucket (0.90); the overall 0.8935 is the summed-confusion
metric, dragged DOWN by n2 (21 eval FP, precision 0.716) and n1 (2 FN). Improving n3
cannot lift the total above n3. The drag is n2 (ΔE+far-field-ARM only, ARM
chance-level → likely structural). Real overall-F1 gains need external info
(imaging / source-volume prior / timing) = part-2 work, not per-event features.

**Choices / caveats:** joint smoothing = 0.5 (was 0 in prod) for sparse-bucket
safety — TUNABLE. n_bins=25 everywhere. `cfg.modeling.double_count_deltaE` is now
MOOT (bucket structure decides counting). Validated logic on synthetic data
(no MEGAlib on Mac, [[no-megalib-on-this-mac]]): summation correct, NaN/empty-class
safe, per-bucket diagnostics work, prior-crush visible in n=1. Real F1 must be run
on the Linux VM. Watch per-bucket TP/FP/FN: expect precision up / recall down in
n=1,2 (1-2-hit signal is only ~5.7% of signal but real). Builds on
[[annihilation-angle-weaknesses]] (the back-to-back score) and the LOR deletion
the deleted LOR feature.
