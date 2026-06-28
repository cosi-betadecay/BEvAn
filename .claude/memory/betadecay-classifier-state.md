---
name: betadecay-classifier-state
description: "Anchor: current state of the COSI β⁺ (positron-annihilation) event classifier — features, bucketed Bayesian model, best eval F1 0.8935 (commit 86bb140), the ~0.89 per-event physics ceiling, and the clean-511 label. Read this first to know where the project stands."
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a4769c0-c4dd-4b11-b3f8-5f862b7c1e69
---

**Task:** classify events in COSI activation simulation (`data/Activation.sim`)
as β⁺ positron-annihilation vs background. Per-event physics features → a
Bayesian likelihood-ratio classifier. Production runs on the Linux MEGAlib VM
(no MEGAlib on this Mac, [[no-megalib-on-this-mac]]); analysis/measurement here
uses `utils/sim_text_reader.py` (pure-Python .sim parser, replicates the label).

**Label (KEEP — clean 511):** `ground_truth_bdecay` / `event_is_bdecay` — signal
iff an ANNI photon's direct secondaries deposit within ~2.87 keV of 511 (3σ of
2.25 FWHM). Broadening to "any detectable annihilation" was tried 2026-06-21 and
REVERTED (would double signal 8.5%→20.2% and redefine the task). Goal as stated is
β⁺ annihilation tagging, but the shipped label is clean-511; revisit only deliberately.

**Shipped features** (`physics/physics_factors.py`):
- `delta_E` — min over reconstructed subsets of |ΣE − 511|.
- `annihilation_angle` — NOT the old min-cos; it's the **vertex-anchored
  back-to-back score** (each hit as candidate vertex, two most anti-parallel arms,
  energy gate ΣarmE>511 to reject single-photon fakes, deficit penalty toward 1022).
  Lower = more annihilation-like. See [[annihilation-angle-weaknesses]].
- `arm` — **REWORKED on branch 75 (2026-06-28):** now a **signed, 511-aware** score
  in `[-1, 1]` using **standard-Compton `theta_kin`** (E_1 = measured total deposit,
  not fixed at 511; the 511 hypothesis was relocated into an energy "push"), selected
  by `argmin|·|`, binned **linear** (was `log,1e-3`). Supersedes the old far-field
  "KNOWN BROKEN / kept-despite-AUC~0.43" ARM. The prior near-field rebuild that cost
  F1 (the reverted near-field ARM rework) is NOT this — this rework WINS (see Best result).
- LOR — DELETED (the deleted LOR feature).

**Architecture:** hit-multiplicity bucketing ([[hit-multiplicity-bucketing]]).
3 buckets routed by feature availability: 1=ΔE, 2=ΔE+ARM, 3=ΔE+ARM+anni. Each
bucket has its OWN class prior + density matrices; per-event likelihood ratio R =
product of its bucket's factors; per-bucket confusion summed for the total.
`_N_BINS={1:25,2:8,3:35}`. Bucket+label assigned together per event (alignment-safe).

**Best result / CHAMPION (UPDATED 2026-06-28, branch 75):** the branch-75 ARM rework
(signed 511-aware ARM + standard-Compton `theta_kin` + linear binning) **is the new
champion — it dominates the old 0.8935 model on EVERY metric** (user-confirmed; exact
new F1 not yet recorded here — fill in from the latest run CSV). This RESOLVES the
"ARM rework unmeasured vs champion" review concern: it was measured and it wins.
The old reference numbers for context: prior champion eval F1 **0.8935** (train ~0.90,
commit `86bb140`); deleted LOR champion 0.8949. ⚠ Memories below that cite 0.8935 as
"the champion" / "the ceiling" predate this and describe the OLD model — the ~0.89
per-event ceiling claim in particular should be re-checked against the new champion.

**Physics ceiling (~0.89, MEASURED):** the residual ~116 FP are ΔE-degenerate,
no-ANNI, multi-hit background — indistinguishable from signal on energy (both 511
AND 1022 keV) and back-to-back geometry; ~61% are genuine non-annihilation, ~39%
were partial-deposit real annihilations the strict-511 label excludes. Per-event
features cannot break this; needs EXTERNAL info (imaging / source-volume prior /
timing) = future work / part-2 paper.

**Open items:** (1) DONE 2026-06-22 (commit dd15978): the entire `tests/` suite was
DELETED (it was stale from the bucketing refactor + LOR deletion and couldn't even
collect). There are now NO tests in the repo; `pytest.ini` is gone too. If tests are
wanted again, write fresh ones against the bucketed dict API (the deleted
scripts/test_population_correction.py @019b278 and scripts/test_tuning.py @e57798f are
recoverable starting points). (2) For
publishability the user is "creating the first β⁺ baseline for COSI" — still needs:
a naïve-floor baseline (ΔE-cut alone), a literature search to position novelty, and
a science figure of merit (not just F1). MEGAlib source for borrowing physics:
`/Users/aryaraeesi/Documents/COSI/megalib` (Revan: src/revan/src/MERCSR*.cxx,
src/global/misc/src/MComptonEvent.cxx).
