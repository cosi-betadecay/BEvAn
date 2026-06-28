---
name: cosi-working-style
description: "How to work on the COSI β⁺ classifier with this user: measure on the parser BEFORE implementing, track cumulative F1 drift across changes, and respect that they want physics-defensibility but will revert if it costs F1."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a4769c0-c4dd-4b11-b3f8-5f862b7c1e69
---

Lessons from the 2026-06-21/22 session (LOR delete → ARM/feature rework → revert):

**Measure before implementing.** The parser (`utils/sim_text_reader.py`, no MEGAlib)
let us kill several plausible-but-wrong ideas cheaply BEFORE coding: Klein-Nishina
as a 2-hit feature (AUC 0.27, anti), tying ARM to the back-to-back vertex (made it
worse), the 1022-keV energy axis separating the residual FP (it didn't). Default to
a quick parser AUC/separation measurement before a pipeline change.
**Why:** saved multiple wasted refactors. **How to apply:** propose → measure on
parser → implement only if it separates.

**Track cumulative F1 drift — don't let principled changes quietly regress.** A run
of individually-reasonable "more physically correct" changes (vertex ARM, 2-hit φ,
4-bucket split, ≥3-merge) dropped eval F1 0.8935→0.8735 step by step; the user got
(rightly) frustrated and we reverted to the peak. I should have flagged the
trajectory after the SECOND down-step, not the third.
**Why:** the user optimizes F1 hard. **How to apply:** after each change, compare to
the running best; if two consecutive changes regress, stop and call it.

**Physics-defensibility vs F1 — they will pick F1 when forced.** The user cares
deeply about physical correctness (deleted the champion LOR feature on physics
grounds; rejected ΔE-only 2-hit as "no physics sense"; insisted ≥3 be ONE stratum
with both matrices into R, not split). BUT when the honest version cost ~0.02 F1
they reverted to the (physically-shaky far-field ARM) peak. Resolution: keep the
*analysis* as paper material even when reverting the *code*.

**At the ceiling, stop micro-optimizing.** The model is at the ~0.89 per-event
ceiling ([[betadecay-classifier-state]]). Further per-event feature/bin tuning just
trades FP↔FN along the Pareto frontier (every change was ~one-for-one). Real gains
need new information (imaging/timing), not more tuning. Push toward a decision
(ship + write up) rather than another tweak.

**The user is sharp on physics** — engage at that level, push back with evidence,
don't hand-wave. They'll catch lazy choices.

**Keep memory for what's load-bearing, not the experiment log (2026-06-28).** The
user wants memory to hold only genuinely important, durable things — NOT every
super-specific one-off (exact F1 deltas of dead-end experiments, stale code pointers,
hyper-narrow single measurements). Bar for saving: would a future session be worse
off without it? Keep — the anchor state, the champion, "don't do X" guardrails,
decisions and their rationale, how-to-work feedback. Drop/fold — micro-findings,
4-decimal numbers from reverted attempts, anything already marked stale.
**Why:** an index bloated with trivia buries the signal. **How to apply:** prefer
folding a finding into an existing entry over a new file; default to NOT saving
unless it clears the bar; when an entry goes stale, prune it rather than appending.
