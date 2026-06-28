# FRETALON (calibration), MELINATOR, REALTA, SPECTRALYZE

The "real hardware" side of MEGAlib: calibrating measured detectors, processing live data,
and identifying spectral lines. This is where the **module/supervisor pipeline** — the most
modern part of the codebase, and the foundation newer COSI tools build on — lives.

---

## FRETALON — calibration framework

Source: `src/fretalon/{base,framework,melinator}`. Three layers.

### base/ — measured-data model
- **`MReadOut`** = one channel's data: a `MReadOutElement` (which channel) + `MReadOutData`
  (the value).
- **`MReadOutElement`** (+ `MReadOutElementStrip`, `MReadOutElementDoubleStrip`) — identifies
  a detector channel (e.g. detector / side / strip).
- **`MReadOutData`** (+ `…ADCValue`, `…Temperature`, `…Timing`, `…WithTiming`, `…Origins`,
  `…Group`) — the measured value(s); a **decorator** chain wraps multiple data types.
- **`MReadOutSequence` / `MReadOutStore`** — collections of read-outs.
- **`MReadOutAssembly`** — the event packet flowing through the pipeline (extends a sequence;
  tracks an analysis-progress bitmask).
- Calibration classes: `MCalibrationSpectrum`, `MCalibrationSpectralPoint`,
  `MCalibrationFitGaussian`/`GaussLandau`, `MCalibrationModel`, and the `MCalibrateEnergy*`
  tasks (find lines → determine model → assign energies). File reader: `MFileReadOuts`
  (`.roa`, see `file-formats.md §6`).

### framework/ — the Module/Supervisor pipeline
A **generic, multi-threaded, queue-based analysis pipeline** — MEGAlib's "module" concept:
- **`MModule`** (abstract) — a self-contained analysis stage. Declares its type, its required
  predecessor/successor module types, and whether it supports multithreading. Lifecycle:
  `Initialize()` → `AnalyzeEvent(MReadOutAssembly*)` loop → `Finalize()`. Modules are wired
  via shared producer/consumer queues, so stages run in parallel.
- Concrete modules: `MModuleLoaderRoa` (read `.roa`), `MModuleSaver` (write
  roa/dat/evta/sim), `MModuleTransmitterRealta` (bridge to REALTA), `MModuleTemplate`
  (boilerplate to copy when writing your own).
- **`MSupervisor`** (singleton) — registers available modules, validates the dependency DAG,
  builds the execution sequence, runs/pauses/interrupts the threaded pipeline, drives the GUI
  (`MGUIExpo`/progress). Config (module list + options) persisted as XML.

**To add a new analysis stage:** copy `MModuleTemplate`, override `AnalyzeEvent()`, declare
dependencies, register with the supervisor, add it to the module Makefile (and ROOT
dictionary — see `conventions-contributing.md`). This is the recommended extension point for
new detector-processing logic (e.g. the COSI detector-effects engine, strip pairing,
calibration application).

### melinator — line/energy calibration GUI
Program: `melinator`. Library uses `MMelinator`. Loads calibration-isotope `.roa` data,
histograms per detector/side/temperature, finds peaks (**Bayesian-block** detection), fits
them (Gaussian / Gauss+Landau with background models), determines a calibration model
(linear / polynomial / spline), and writes an energy-calibration `.ecal`.

```bash
melinator -l                 # load last calibration files into the GUI
melinator -p                 # load last + parametrize all peaks
melinator -a -e Cal.ecal     # auto: load, parametrize, save, exit
melinator -d 1 -s 0          # restrict to detector 1, side 0
```
Confirm flags in `src/fretalon/melinator/src/MInterfaceMelinator.cxx::Usage()`.

---

## REALTA — real-time analysis

Program: `realta`. Library: `libRealta`. Source: `src/realta/{inc,src}`. Runs the
reconstruction+imaging pipeline **live** on a network data stream (e.g. a detector during a
balloon/space campaign).

- **`MRealTimeAnalyzer`** runs ~7 parallel threads connected by queues: Transmission
  (receive `MHitEvent` over TCP/IP) → Coincidence → Reconstruction (`MRawEventAnalyzer` →
  `MComptonEvent`) → Imaging (`MImage` sky maps) → Histogramming (count-rate, spectra) →
  Identification (`MSpectralAnalyzer` isotope ID) → CleanUp (purge old events by accumulation
  window).
- Networking: `MTransceiverTcpIp` (+ ROOT `TSocket`), `MRealTimeEvent` carries an event
  through the stages with state flags (initialized/coincident/reconstructed/imaged/dropped).
- GUI panels: `MGUIControlCenter`, `MGUINetwork`, `MGUIAccumulation`. Config:
  `MSettingsRealta` (host, port, geometry, thresholds, accumulation time).

It's a GUI application; launch `realta` and configure the network source. Each thread keeps
its own geometry copy to avoid lock contention.

---

## SPECTRALYZE — spectral line / isotope identification

Program: `spectralyze`. Library: `libSpectralyze`. Source: `src/spectralyze/{inc,src}`.
Small module (~2k lines).

- **`MSpectralAnalyzer`** — set a spectrum (`TH1D` or bins/range + binning mode:
  linear/log/variable-by-resolution), find peaks (ROOT `TSpectrum::Search` or Bayesian
  blocks), fit each peak (Gaussian/Gauss-Landau + background), then match peak energies to a
  loaded isotope line database.
- **`MPeak`** — fitted peak (centroid, sigma, counts, background, significance S/√B, candidate
  isotopes). **`MQualifiedIsotope`** — isotope with decay mode and known lines.
- Used standalone and inside REALTA's identification thread. Isotope libraries live in
  `resource/libraries/*.isotopes`.
