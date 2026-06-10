---
name: megalib
description: MEGAlib (Medium Energy Gamma-ray Astronomy library) reference for this repo — .sim/.tra file formats, IA/ANNI/HTsim line field maps, particle ID scheme, the ROOT Python bindings (MFileEventsSim, MSimEvent, MSimIA, MSimHT), cosima/revan/geomega usage, and how this project's pipeline consumes them. Use when parsing or reasoning about data/Activation.sim or .tra files, extracting ANNI ground truth, working with reader_extraction.py / ground_truths.py / event_processing.py, or running cosima/revan/mimrec.
---

# MEGAlib for betadecay-analysis

MEGAlib is the COSI collaboration's Geant4-based simulation + analysis
suite. **Full source is local at `/Users/aryaraeesi/Documents/COSI/megalib`**
(src/, doc/ with official PDF manuals, resource/examples/) — when a format
or API question isn't answered here, grep the source; the parsing code in
`src/sivan/src/` is ground truth.

Toolchain: **cosima** (simulation → `.sim`), **revan** (reconstruction
`.sim` → `.tra`), **mimrec** (imaging), **geomega** (geometry). This repo
reads the `.sim` file directly through PyROOT bindings.

## Reference files (read on demand)

- `references/sim-format.md` — authoritative .sim spec: header, SE block
  keywords, the 24-field IA line map, ANNI ground-truth semantics, particle
  IDs (source-verified — the Cosima.pdf table is wrong for IDs ≥ 4), HTsim.
- `references/python-api.md` — PyROOT API for MFileEventsSim, MSimEvent,
  MSimIA, MSimHT, MVector, MDGeometryQuest, MComptonEvent (+ Python gotchas).
- `references/revan-tra.md` — revan CLI/pipeline, .tra format, sim↔tra ID
  mapping (tra IDs are a gappy subset; join on ID, never positionally).
- `references/cosima-geomega.md` — .source files (incl. how to author a
  clean β⁺ point-source validation sim), the 3-step activation workflow,
  Max.geo.setup summary, detector type IDs.

## This repo's dataset (data/Activation.sim)

- 119 MB, MEGAlib 4.00.00. `BeamType Activation`: decays of activated
  nuclei (Ge-71, Cu-61, ... — nuclei encode as 1000·Z+A) in a 4-layer
  Strip3D germanium stack (Max.geo.setup; all hits are detector type 3).
- **52,932 events**, **121,838 hits**, **25,130 IA ANNI lines = 12,565
  ground-truth annihilations** (2 lines per annihilation: same vertex,
  exactly negated photon unit directions, ~511 keV each; in-flight
  annihilations exist and break both rules).
- Signal and background both come from this one file; labeling is per
  event: `ground_truth_bdecay` = any ANNI whose secondaries deposit hit
  energy within 3σ-FWHM tolerance of 511 keV (`physics/ground_truths.py`).
- Units: cm, keV, seconds.

## How the pipeline consumes it

- `utils/reader_extraction.py` — opens sim+tra readers (`MFileEventsSim`,
  `MFileEventsTra`), auto-runs `revan -g <geo> -f <sim> -n -a` if the .tra
  is missing. Needs the `MEGALIB` env var (config:
  `${oc.env:MEGALIB}/resource/examples/geomega/special/Max.geo.setup`).
- `physics/event_processing.py` — hits → (energies, positions, sizes)
  tensors + subset/ordering enumeration (7-hit subset cap).
- `dataset/datasets.py` — iterates sim events, computes delta_E /
  annihilation_angle / arm, splits bdecay vs bg by ground truth.
- NaN convention: feature NaN → density 0 downstream → classified bg.
- Indexing gotcha: `GetIAAt(i)`/`GetHTAt(i)` are 0-based; IA IDs and
  OriginIDs are 1-based (`process_id = i + 1` in ground_truths.py).

## Operational gotchas

- **Flaky ROOT teardown segfault** after eval output prints
  (`gSystem.ProcessEvents` in a background thread). Results already
  printed are valid; re-run before treating a nonzero exit as real
  (SCOREBOARD_V1 iter 8).
- revan with no `-c` reads `~/.revan.cfg` if present → reconstruction
  params are machine-state. Pin a config for reproducibility.
- The geometry path inside the .sim header is from the generating machine;
  always pass the local geometry explicitly.
- The .sim is text: for format questions, `grep`/`awk` it directly instead
  of loading through ROOT (~minutes for a full PyROOT pass).
- ANNI truth (vertex + photon axis) is simulation-only metadata — valid
  for development/validation metrics, never as a runtime feature input.
