# MEGAlib Architecture

## What MEGAlib is

A toolkit for the complete analysis chain of Compton telescopes and hard X-/gamma-ray
detectors (originally astrophysics; also medical imaging, environmental monitoring,
homeland security). Built on **ROOT** and **Geant4**. ~311k lines of C++ across ~1000
files in `src/`. Single principal author (Andreas Zoglauer) plus contributors; the style
is consistent and idiomatic-to-itself but predates modern C++ idioms in many places.

## The data-flow pipeline

The modules correspond to physical analysis steps. Data formats are the contracts between
them (see `file-formats.md`):

```
 geometry        simulation        event recon        image recon
 .geo.setup  →   .source → .sim →  .sim/.evta → .tra → .tra → image / ARM / spectrum
  GEOMEGA          COSIMA            REVAN              MIMREC
```

- **GEOMEGA** defines the instrument (volumes, materials, detectors, triggers). Every
  other module loads a geometry to know detector positions, resolutions, and thresholds.
- **COSIMA** runs Geant4 using that geometry and a `.source` description, producing `.sim`
  (every simulated interaction + truth).
- **SIVAN** analyzes `.sim` against truth (efficiency, energy containment, etc.).
- **REVAN** turns raw hits (`.sim` for simulations, `.evta` for measured data) into
  reconstructed physical events (`.tra`): clustering → electron tracking → Compton
  sequence reconstruction.
- **MIMREC** images `.tra` events (list-mode maximum-likelihood deconvolution) and does
  high-level analysis: spectra, ARM, polarization, sensitivity, light curves.
- **RESPONSE** builds the instrument response matrices (`.rsp`) by matching SIVAN truth to
  REVAN reconstruction; these feed both REVAN (Bayesian sequencing) and MIMREC (imaging).
- **FRETALON** calibrates real hardware (raw `.roa` read-outs → energy calibration
  `.ecal`), via a generic module/supervisor pipeline; **melinator** is its line-calibration GUI.
- **REALTA** runs the reconstruction+imaging pipeline live on a network stream.
- **SPECTRALYZE** does spectral line finding / isotope identification.

## Module dependency order

Libraries build bottom-up (see top-level `Makefile` target ordering):

```
global/misc (libCommonMisc)  ← foundation, everything depends on it
global/gui  (libCommonGui)
global/neuralnet
   ↓
geomega (libGeomega/+Gui)    ← geometry, depended on by nearly everything
   ↓
sivan, revan, mimrec, spectralyze, response, fretalon, realta, eview
   ↓
cosima (links Geant4)        ← simulation
   ↓
addon/*                      ← standalone tools link many of the above
```

A combined `libMEGAlib.so` is produced (`make combine`) bundling all object files; this
is what PyROOT/Julia load.

## Core abstractions you will see everywhere

- **`MString`** — the string type (wraps `std::string`, adds tokenizing, conversions).
- **`MVector`** — 3D vector with Cartesian↔spherical; `MRotation`, `MQuaternion` for orientation.
- **`MTokenizer` / `MParser`** — line-based parsing of MEGAlib's ASCII formats, with an
  in-brace `{...}` math evaluator. All `.geo.setup`/`.source`/`.sim`/`.tra` reading funnels
  through these.
- **`MFile`** and the **`MFileEvents*`** family — transparent gzip + ASCII/binary readers
  for each event format (`MFileEventsSim` for `.sim`, `MFileEventsTra` for `.tra`,
  `MFileEventsEvta` for `.evta`).
- **`MPhysicalEvent`** hierarchy — the reconstructed event types:
  `MComptonEvent`, `MPairEvent`, `MPhotoEvent`, `MPETEvent`, `MMuonEvent`, `MDecayEvent`,
  `MMultiEvent`, `MUnidentifiableEvent`. This is the universal "what interaction happened".
- **`MResponseMatrixO1..O17` / `MResponseMatrixON`** — N-dimensional histograms; the central
  data structure tying simulation to imaging/sequencing. (`MResponseMatrixAxis`,
  `MResponseMatrixAxisSpheric` for FISBEL spherical binning.)
- **`MDGeometryQuest`** — the queryable geometry (volume/material at a point, noising,
  resolutions). Subclassed per module (`MGeometryRevan`).
- **`MStreams`** (`mout`, `merr`, `mlog`, `mdebug`, `massert`, `mgui`) — logging facade.
- **`MSettings*`** — XML config persistence (the `.cfg` files).
- **`MModule` / `MSupervisor`** — the newer (FRETALON) generic, multi-threaded, queue-based
  analysis-pipeline framework; the direction newer COSI tooling builds on.

## ROOT and Geant4 coupling

- **ROOT** is pervasive and mostly non-optional: `TH1/TH2/TH3` histograms, `TF1`/`TMinuit`
  fitting, `TGeoManager` geometry, `TMVA` machine-learning reconstruction, `TGFrame` GUI,
  `TThread`/`TMutex` threading, and **ROOT dictionaries** (`ClassDef`/`ClassImp` +
  `rootcling`) for serialization and PyROOT reflection. ~140 classes register dictionaries.
- **Geant4** is used only by **cosima**. `MCPhysicsList` selects EM (Standard/Livermore/
  LivermorePol/Penelope) and hadronic (QGSP/FTFP variants) physics; `MCDetectorConstruction`
  converts the Geomega geometry into Geant4 volumes.
- Build-time **version gating**: `config/AllowedROOTVersions.txt` and
  `AllowedGeant4Versions.txt` restrict which dependency versions are accepted.

## Cross-cutting patterns & idioms

- Naming: classes `M<Name>`, GUI `MGUI<Name>`, members `m_UpperCamelCase`, locals
  `UpperCamelCase`, Doxygen `//!` comments. (See `conventions-contributing.md`.)
- Heavy **strategy/inheritance** families: many interchangeable algorithms behind a base
  class (e.g. `MERCSR*` Compton reconstruction, `MERTrack*` tracking, `MResponse*`,
  `MBackprojection*`, `MBinner*`).
- **Manual memory management** (`new`/`delete`, raw pointers, no smart pointers, little
  RAII). Events and matrices are owned-by-caller; when looping events from a reader you
  must `delete` each (in PyROOT: `M.SetOwnership(Event, True)`).
- Hand-written ASCII parsers and binary stores (`MBinaryStore`) rather than a serialization
  library.
- Multi-threading via ROOT `TThread`+`TMutex`; most data classes are *not* themselves thread
  safe, so threaded code keeps per-thread geometry copies.

## Known technical debt (be aware when editing)

- No smart pointers / RAII → exception-unsafe paths, leak risk; respect existing
  ownership conventions rather than "modernizing" piecemeal.
- `using namespace std;` in headers; global/singleton state (`gGeoManager`, version
  globals, `MException::m_Abort`).
- Several god objects: `MRawEventAnalyzer`, `MDGeometry`, `MCSource`, `MPhysicalEvent`.
- Hardcoded scaling limits (response-matrix orders fixed O1–O17; per-detector params D1–D8).
- Thin automated testing (a few unit tests under `resource/examples/unittests/`; CI is
  **CodeQL static analysis only**, no build/test in GitHub Actions). Correctness leans on
  the worked examples in `resource/examples/`.
- A few explicitly deprecated-but-live methods (e.g. `MSimEventLoader`, `MEventTransmitter`).

## Where things live

```
src/global/{misc,gui,neuralnet}   foundation libraries
src/geomega/{inc,src}             geometry
src/cosima/{inc,src,unittests}    Geant4 simulation (.hh/.cc, not .h/.cxx)
src/revan/{inc,src}               event reconstruction
src/mimrec/{inc,src}              imaging + high-level analysis
src/sivan/{inc,src}               simulated-event analysis
src/response/{inc,src}            response-matrix generation
src/fretalon/{base,framework,melinator}   calibration framework
src/realta/{inc,src}              real-time analysis
src/spectralyze/{inc,src}         spectral analysis
src/eview/{inc,src}               event viewer
src/addon/*                       standalone tools (each its own program)
bin/                              wrapper scripts + symlinks to built programs
config/                          configure machinery, Makefile templates, version gates
resource/examples/               worked examples (THE practical documentation)
resource/geometries, catalogs, patches, libraries
doc/                             PDFs (MEGAlib, Cosima, Geomega, Mimrec) + ChangeLog
```

Note the **cosima exception**: it uses Geant4-style extensions `.hh`/`.cc`, while the rest
of MEGAlib uses `.h`/`.cxx`.
