---
name: megalib
description: >-
  MEGAlib (Medium-Energy Gamma-ray Astronomy library) reference for the
  betadecay-analysis repo AND a comprehensive guide to MEGAlib as a whole. Covers
  this project's pipeline (parsing data/Activation.sim and .tra, extracting ANNI/511
  annihilation ground truth, reader_extraction.py / ground_truths.py /
  event_processing.py, the PyROOT bindings MFileEventsSim/MSimEvent/MSimIA/MSimHT) plus
  the full MEGAlib toolkit — its programs (cosima, revan, mimrec, geomega, sivan,
  responsecreator, melinator, realta, spectralyze), file formats (.geo.setup, .source,
  .sim, .tra, .evta, .roa, .rsp, .cfg), M-prefixed C++ classes, build system, and
  response/calibration/imaging workflows. Use when parsing or reasoning about .sim/.tra
  files, running cosima/revan/mimrec/geomega, building geometries or response matrices,
  calibrating detectors, or anything MEGAlib.
---

# MEGAlib for betadecay-analysis (and the full MEGAlib reference)

MEGAlib is the COSI collaboration's Geant4-based simulation + analysis suite for Compton
telescopes and hard X-/gamma-ray detectors. **Full source is local at
`/Users/aryaraeesi/Documents/COSI/megalib`** (`src/`, `doc/` with official PDF manuals,
`resource/examples/`) — when a format or API question isn't answered here, grep the source;
the parsing code in `src/sivan/src/` is ground truth.

Toolchain pipeline: **geomega** (geometry `.geo.setup`) → **cosima** (simulation → `.sim`)
→ **revan** (reconstruction `.sim`/`.evta` → `.tra`) → **mimrec** (imaging + ARM / spectra
/ polarization). Plus **responsecreator** (response matrices `.rsp`), **fretalon/melinator**
(detector calibration), **sivan** (simulated-truth analysis), **realta** (real-time),
**spectralyze** (line/isotope ID). This repo reads the `.sim` (and reconstructed `.tra`)
directly through PyROOT bindings.

## Two tiers of reference — read the project tier first

### Tier 1 — project-specific & source-verified (authoritative for THIS repo's data/formats)
These are line-cited against the local source and tuned to `data/Activation.sim`. **When
they overlap a general file below, they win.**

- `references/sim-format.md` — authoritative `.sim` spec: header, SE-block keywords, the
  **24-field IA line map**, ANNI ground-truth semantics, the **particle-ID table**
  (source-verified — the Cosima.pdf table is wrong for IDs ≥ 4; **2 = e⁺, 3 = e⁻**), HTsim.
- `references/python-api.md` — PyROOT API for `MFileEventsSim`, `MSimEvent`, `MSimIA`,
  `MSimHT`, `MVector`, `MDGeometryQuest`, `MFileEventsTra`, `MComptonEvent` (+ Python gotchas).
- `references/revan-tra.md` — revan CLI/pipeline, `.tra` format, **event-type codes**
  (`CO/PH/PA/MU/DY/PT/MT/UN`), and the **sim↔tra ID mapping** (tra IDs are a gappy subset;
  join on `ID`, never positionally).
- `references/cosima-geomega.md` — `.source` files (incl. authoring a clean β⁺ point-source
  validation sim), the 3-step activation workflow, `Max.geo.setup` summary, **detector-type
  IDs** (`3 = Strip3D` = this data).

### Tier 2 — general MEGAlib reference (broad coverage of the whole toolkit)
For modules/tools/workflows beyond this repo's immediate `.sim` parsing.

- `references/architecture.md` — overall design, the full pipeline, module dependencies,
  cross-cutting patterns, known tech debt.
- `references/file-formats.md` — all formats overview (`.geo.setup`, `.source`, `.evta`,
  `.roa`, `.rsp`, `.cfg`); for `.sim`/`.tra` it defers to the Tier-1 files.
- `references/geomega-geometry.md` — geometry model, all detector types, shapes, triggers.
- `references/cosima-simulation.md` — general cosima: physics lists, all source/beam/spectrum
  types, activation, parallel/distributed running.
- `references/revan-reconstruction.md` — reconstruction algorithms in depth (clustering,
  tracking, Compton sequence reconstruction variants).
- `references/mimrec-imaging.md` — list-mode ML imaging (EM/OSEM), ARM, spectra, polarization,
  sensitivity, event selection.
- `references/sivan-response.md` — sivan truth analysis + **response-matrix generation**
  (`responsecreator` modes) — the part this repo's skill didn't previously cover.
- `references/calibration-realtime.md` — FRETALON module/supervisor framework, melinator,
  realta, spectralyze.
- `references/tools-scripts.md` — all `bin/` scripts, distributed computing (mcosima/dcosima/
  dmegalib), addon tools, the PyROOT/Julia binding pattern.
- `references/global-api.md` — the foundation C++ classes broadly (defers to python-api.md for
  this repo's PyROOT usage).
- `references/install-build.md` — setup.sh, configure/make, ROOT/Geant4 versions, Docker,
  ROOT dictionaries.
- `references/conventions-contributing.md` — coding style, `.clang-format`, adding a class +
  dictionary, CI.
- `references/workflows.md` — end-to-end recipes (sim→recon→image, build a response, ARM,
  calibration, PyROOT loop, cluster runs).

## This repo's dataset (data/Activation.sim)

- 119 MB, MEGAlib 4.00.00. `BeamType Activation`: decays of activated nuclei (Ge-71, Cu-61,
  … — nuclei encode as 1000·Z+A) in a 4-layer Strip3D germanium stack (`Max.geo.setup`; all
  hits are detector type 3).
- **52,932 events**, **121,838 hits**, **25,130 IA ANNI lines = 12,565 ground-truth
  annihilations** (2 lines per annihilation: same vertex, exactly negated photon unit
  directions, ~511 keV each; in-flight annihilations exist and break both rules).
- Signal and background both come from this one file; labeling is per event:
  `ground_truth_bdecay` = any ANNI whose secondaries deposit hit energy within 3σ-FWHM
  tolerance of 511 keV (`physics/ground_truths.py`).
- Units: cm, keV, seconds.

## How the pipeline consumes it

- `utils/reader_extraction.py` — opens sim+tra readers (`MFileEventsSim`, `MFileEventsTra`),
  auto-runs `revan -g <geo> -f <sim> -n -a` if the `.tra` is missing. Needs the `MEGALIB`
  env var (config: `${oc.env:MEGALIB}/resource/examples/geomega/special/Max.geo.setup`).
- `physics/event_processing.py` — hits → (energies, positions, sizes) tensors +
  subset/ordering enumeration (7-hit subset cap).
- `dataset/datasets.py` — iterates sim events, computes delta_E / annihilation_angle / arm,
  splits bdecay vs bg by ground truth.
- NaN convention: feature NaN → density 0 downstream → classified bg.
- Indexing gotcha: `GetIAAt(i)`/`GetHTAt(i)` are 0-based; IA IDs and OriginIDs are 1-based
  (`process_id = i + 1` in ground_truths.py).

## Operational gotchas

- **Flaky ROOT teardown segfault** after eval output prints (`gSystem.ProcessEvents` in a
  background thread). Results already printed are valid; re-run before treating a nonzero
  exit as real (SCOREBOARD_V1 iter 8).
- revan with no `-c` reads `~/.revan.cfg` if present → reconstruction params are
  machine-state. Pin a config for reproducibility.
- The geometry path inside the `.sim` header is from the generating machine; always pass the
  local geometry explicitly.
- The `.sim` is text: for format questions, `grep`/`awk` it directly instead of loading
  through ROOT (~minutes for a full PyROOT pass).
- ANNI truth (vertex + photon axis) is simulation-only metadata — valid for
  development/validation metrics, never as a runtime feature input.

## Module map (the whole toolkit)

| Module | Program(s) | Class prefix | Tier-1 file | Tier-2 file |
|--------|-----------|--------------|-------------|-------------|
| Foundation libs | — | `M*`, `MGUI*` | python-api.md | global-api.md |
| Geometry | `geomega` | `MD*` | cosima-geomega.md | geomega-geometry.md |
| Simulation (Geant4) | `cosima` | `MC*` | cosima-geomega.md, sim-format.md | cosima-simulation.md |
| Event reconstruction | `revan` | `MER*`, `MRE*` | revan-tra.md | revan-reconstruction.md |
| Image reconstruction | `mimrec` | `MI*` | — | mimrec-imaging.md |
| Simulated-truth analysis | `sivan` | `MSim*` | sim-format.md, python-api.md | sivan-response.md |
| Response generation | `responsecreator` | `MResponse*` | — | sivan-response.md |
| Calibration / real-time / spectra | `melinator`, `realta`, `spectralyze` | `MModule`, … | — | calibration-realtime.md |
| Tools, scripts, bindings | `bin/*`, `src/addon/*` | — | — | tools-scripts.md |
