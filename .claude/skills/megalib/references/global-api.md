# Foundation API (global) — the classes you use everywhere

`src/global/{misc,gui,neuralnet}`. These are the reusable building blocks under every
module, usable from C++ and (via PyROOT) Python/Julia. Headers in `src/global/misc/inc/`.
Grep for a class then read its header — they're the best documentation.

> **For this repo's PyROOT usage, `references/python-api.md` is authoritative** — it has the
> exact method signatures and Python gotchas for the classes this project actually uses
> (`MFileEventsSim`, `MSimEvent`, `MSimIA`, `MSimHT`, `MVector`, `MDGeometryQuest`,
> `MFileEventsTra`, `MComptonEvent`). This file is the broader C++-foundation overview.

## Strings, parsing, math

- **`MString`** — the string type. `std::string` wrapper + `Tokenize`, `Contains`,
  `BeginsWith`/`EndsWith`, `ReplaceAll`, `ToInt/ToDouble`, scientific rounding. Public APIs
  take/return `MString`; from Python wrap literals: `M.MString("...")`.
- **`MTokenizer`** — splits a line into typed tokens; `GetTokenAtAsDouble/Int/String/...`;
  evaluates `{ ... }` math expressions and composed `a.b` tokens. The engine behind all
  ASCII-format reading.
- **`MParser`** (extends `MFile`) — line-by-line tokenized file base class; subclass for new
  ASCII formats.
- **`MVector`** — 3D vector; Cartesian (`X/Y/Z`) ↔ spherical (`Mag/Theta/Phi`), `Dot`,
  `Cross`, `Angle`, `RotateX/Y/Z`, `RotateAroundVector`. The workhorse for positions/directions.
- **`MRotation`** (3×3) and **`MQuaternion`** — orientations; `MRotationInterface` is mixed
  into events for pointing.
- **`MMath`** (coordinate conversions, stats tests), **`MFastMath`** (approx atan/atan2 for
  hot loops).
- **`MTime`** — nanosecond timestamps, MEGAlib `TI <sec.nsec>` format, arithmetic.

## Files & events

- **`MFile`** — base file wrapper: transparent gzip, ASCII/binary, append, progress, seek.
- **`MBinaryStore`** — typed binary (de)serialization (`AddInt32/GetDouble/...`).
- **`MFileEvents`** + subclasses — the event-file readers/writers:
  - `MFileEventsSim` → `.sim` (yields `MSimEvent`)
  - `MFileEventsTra` → `.tra` (yields `MPhysicalEvent`)
  - `MFileEventsEvta` → `.evta`
  They handle the `IN`(include)/`NF`(next-file) chaining and observation time.
- **Event loop pattern** (C++ and PyROOT): `Open()` → loop `GetNextEvent()` until null →
  `Close()`. **You own each returned event — `delete` it (C++) / `M.SetOwnership(ev, True)`
  (Python)** or you leak.

## Physical events (the `.tra` content)

`MPhysicalEvent` base (carries id, time, pointing) + subclasses:
`MComptonEvent` (scatter energies/positions → Compton cone), `MPairEvent`, `MPhotoEvent`,
`MPETEvent`, `MMuonEvent`, `MDecayEvent`, `MMultiEvent`, `MUnidentifiableEvent`. Key getters:
`GetEnergy()`, `GetPosition()`, `GetOrigin()` (incoming direction), event-type predicates,
`ParseLine`/`Stream` for I/O, `Assimilate`/`Duplicate`.

## Response matrices

`MResponseMatrixON` (generic N-D) and fixed-order `MResponseMatrixO1..O17`
(+ `MResponseMatrixOx` base, `MResponseMatrixAxis`/`…AxisSpheric` for FISBEL spherical bins).
`Read`/`Write` `.rsp`, `Add`, `Get`/`GetInterpolated`, `GetAxis*`, sparse/dense modes,
`Collapse`, arithmetic operators. Created by `responsecreator`; consumed by mimrec/revan.

## Geometry access

`MDGeometryQuest` (see `geomega-geometry.md`) — `ScanSetupFile()`, then query
volume/material/sensitivity at a point, noise hits, fetch resolutions. The handle you pass
to event readers and analyzers.

## Functions, binning, images

- `MFunction` / `MFunction2D` / `MFunction3D` / `MFunction3DSpherical` — interpolated
  functions loaded from data (linear/log), with `Evaluate`, `Integrate`, `GetRandom`
  (used for spectra, profiles, light curves).
- `MBinner*` — binning strategies: fixed number, fixed counts/bin, **Bayesian blocks**,
  **FISBEL** (spherical).
- `MImage` / `MImage2D` / `MImage3D` (+ `…Update`) — ROOT-`TH*`-backed display with color
  palettes/draw options; `MImageSpheric`/`MImageGalactic` for sky projections.

## Logging, errors, settings

- **Streams** (`MStreams.h`): use `mout` (stdout), `merr`, `mlog`, `mdebug`, `mgui`
  (GUI log), and `massert(cond)` for assertions. Thread-safe via `TMutex`. This is how
  MEGAlib code reports — prefer it over raw `std::cout`.
- **`MExceptions`** — `MException` hierarchy (`MExceptionParameterOutOfRange`,
  `IndexOutOfBounds`, `ValueNotFound`, `DivisionByZero`, `NumberNotFinite`, …); a static
  abort flag controls fatal behavior.
- **`MSettings*` / `MXmlDocument` / `MXmlNode`** — the `.cfg` XML config layer.
- **`MGUI*`** (in `global/gui`) — ROOT `TGFrame`-based GUI toolkit; `MGUIMain`,
  `MGUIDialog`, and ~30 `MGUIE*` widgets. Base classes for every module's GUI.

## Neural nets

`global/neuralnet`: a small feed-forward backprop library (`MNeuralNetwork(Backpropagation)`,
`MNeuron`, `MSynapse`). Used by some reconstruction variants; most ML now goes through ROOT
**TMVA** instead.

## Initialization

Most standalone/PyROOT use starts with `MGlobal::Initialize()` (loads the library, sets up
ROOT). Compile-time flags in `MGlobal.h` (`USEROOT`, `USEGUI`, version constants).
