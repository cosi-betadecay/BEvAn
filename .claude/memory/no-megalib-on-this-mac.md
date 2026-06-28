---
name: no-megalib-on-this-mac
description: This Mac cannot run PyROOT/MEGAlib — use the pure-Python sim text reader; production runs happen on a Linux machine
metadata: 
  node_type: memory
  type: project
  originSessionId: 8ec9ec2e-3b87-4d91-ae86-08bdbc90ce6b
---

This Mac has no MEGAlib build (no libMEGAlib, no revan/cosima binaries) and
homebrew ROOT 6.36 is built for Python 3.14 while the project venv is 3.10 —
`import ROOT` fails everywhere. The RL loops / production pipeline ran on a
Linux machine (the .tra header path says /home/arya/...).

**How to apply:** for anything needing event data on this machine, use
`utils/sim_text_reader.py` (pure-Python .sim parser, full file in ~3 s) —
never attempt the PyROOT path here. `tests_specialized/conftest.py` imports
ROOT lazily inside `real_tensors` so collection works; keep it that way.
The the deleted LOR feature harness is built on the text reader for this
reason. MEGAlib source for format/API questions: ~/Documents/COSI/megalib.

**CRITICAL CAVEAT — the parser is NOT faithful on ENERGIES (proven on VM 2026-06-22).**
`sim_text_reader.py` reads the raw `.sim` HTsim energy lines = UN-NOISED. Production
opens the .sim with `MFileEventsSim`, which APPLIES detector energy smearing by
default (σ≈0.955 keV/hit). VM gate-check: true-signal `delta_E` median = **0.460 keV
in production vs 0.001 keV in the parser** (frac<0.2keV: 23.7% vs 99.6%). So any
ENERGY-quantitative parser measurement (delta_E shape, ΔE-degenerate populations,
energy-partition/CKD AUCs) does NOT transfer to production and can be flat wrong —
this is exactly what produced the refuted "delta_E-only 0.99" claim
([[exhausted-levers]]). Positions are likely also smeared by MFileEventsSim,
so treat geometry magnitudes as suspect too. Parser is reliable for LABELS and
STRUCTURE (hit counts, origin tracking, event topology), NOT for noised observables.
Verify any energy/geometry-feature magnitude on the VM before trusting it.
