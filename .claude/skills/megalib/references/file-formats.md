# MEGAlib file formats

The most load-bearing reference. MEGAlib uses **hand-written, line-oriented ASCII parsers**
(via `MTokenizer`/`MParser`), so **keyword spelling, order, and argument counts matter**.
When unsure of an exact keyword, grep the parser:
`grep -rn "KEYWORD" $MEGALIB/src/<module>/src`. Real worked files live under
`resource/examples/`.

Quick map of extensions:

| Ext | Contents | Produced by | Consumed by | Parser source |
|-----|----------|-------------|-------------|---------------|
| `.geo.setup` `.geo` | geometry/detector model | hand-written | all modules | `src/geomega/src/MDGeometry.cxx::ScanSetupFile` |
| `.source` | simulation run/source description | hand-written | cosima | `src/cosima/src/MCParameterFile.cxx`, `MCSource.cxx` |
| `.sim` `.sim.gz` `.sim.bin` | simulated events (every interaction + truth) | cosima | sivan, revan, response | `src/sivan/src/MFileEventsSim.cxx`, `MSimEvent.cxx` |
| `.evta` | calibrated/measured raw hits | hardware/fretalon | revan | `src/global/misc/src/MFileEventsEvta.cxx` |
| `.tra` `.tra.gz` | reconstructed physical events | revan | mimrec, eview | `MFileEventsTra.cxx`, `MPhysicalEvent::ParseLine` |
| `.roa` | raw read-outs (ADC etc.) | hardware | fretalon/melinator | `src/fretalon/base/src/MFileReadOuts.cxx` |
| `.rsp` `.rsp.gz` | N-dim response matrix | responsecreator | mimrec, revan | `src/global/misc/src/MResponseMatrixON*.cxx` |
| `.cfg` (`.revan.cfg`, `.mimrec.cfg`, …) | XML app configuration | GUIs | revan/mimrec/… | `MSettings*` (XML) |
| `.ecal` | energy calibration | melinator | revan/fretalon | fretalon |

General syntax rules for the ASCII (non-`.cfg`) formats:
- One directive per line; tokens whitespace-separated.
- `#` starts a comment; `//` is also tolerated in `.geo.setup`.
- `$(MEGALIB)` / `${MEGALIB}` env-var expansion is supported in paths.
- `.geo.setup`/`.source` support `Include <file>` and `Constant <name> <value>`; constants
  and `{ ... }` math expressions can be used in numeric fields.
- Detectors/sources use a **dotted property syntax**: declare a named object, then set
  `Name.Property value` lines.

---

## 1. `.geo.setup` — geometry

Defines the world volume tree, materials, detectors, triggers. Verified example
(`resource/examples/benchmarks/Benchmark.geo.setup`):

```text
Name Benchmark                              # geometry name
Version 2.0

SurroundingSphere 10  0.0 0.0 0.0  10.0    # Radius  X Y Z(center)  Distance — the sim "sky"

Include $(MEGALIB)/resource/examples/geomega/materials/Materials.geo   # pull in materials

# ---- Volume tree ----
Volume WorldVolume                         # the top volume MUST exist and be named WorldVolume
WorldVolume.Material Vacuum
WorldVolume.Visibility 0                   # 0 invisible, 1 visible
WorldVolume.Shape BRIK 5000. 5000. 5000.   # box half-extents (cm)
WorldVolume.Mother 0                        # 0 == world has no mother

Volume GeWafer
GeWafer.Material Germanium
GeWafer.Shape BOX 4.0 4.0 0.5              # BOX == BRIK (half-extents cm)
GeWafer.Position 0 0 0                      # X Y Z in mother frame (cm)
GeWafer.Mother WorldVolume

# ---- Detector + trigger ----
MDStrip3D                  ActiveDetector   # <DetectorType> <DetectorName>
ActiveDetector.SensitiveVolume   GeWafer    # which volume is active
ActiveDetector.Offset            0.3 0.3    # dead edge / strip offset (cm)
ActiveDetector.StripNumber       40 40      # strips in x, y
ActiveDetector.NoiseThreshold    15         # keV — below this no hit recorded
ActiveDetector.TriggerThreshold  30         # keV — energy to raise a trigger
ActiveDetector.DepthResolution   10  0.02   # at E=10 keV, depth sigma = 0.02 cm (repeat per E)
ActiveDetector.EnergyResolution Gaus 100 100 0.454   # type Ein Epeak sigma(keV)  (repeat per E)
ActiveDetector.GuardringEnergyResolution 1000 1.5
ActiveDetector.GuardringTriggerThreshold 40

Trigger ActiveDetectorTrigger
ActiveDetectorTrigger.Veto false                 # false = this is a trigger condition
ActiveDetectorTrigger.TriggerByDetector true
ActiveDetectorTrigger.Detector ActiveDetector 1  # require >=1 hit in this detector

Trigger Detector7_GRTrigger
Detector7_GRTrigger.Veto true                    # true = veto (reject) condition
Detector7_GRTrigger.TriggerByDetector True
Detector7_GRTrigger.GuardringDetector ActiveDetector 1
```

Header/global keywords: `Name`, `Version`, `SurroundingSphere R X Y Z D`,
`ShowSurroundingSphere`, `Include`, `Constant`, `Vector`, `Material` (see materials file),
`DoSanityChecks`, `VirtualizeNonDetectorVolumes`, `DetectorSearchTolerance`,
`CrossSectionFileDirectory`.

Volume properties: `.Material`, `.Shape <TYPE ...>`, `.Position x y z`,
`.Rotation` (several conventions; check existing geometry — Euler angles in degrees),
`.Mother`, `.Visibility`, `.Color`, `.LineWidth`, `.Many`/`.Copy` (cloned placements),
`.Virtual`.

Shapes (map to ROOT `TGeo*`): `BRIK`/`BOX x y z`, `TUBS rmin rmax dz phi1 phi2`,
`SPHE rmin rmax theta1 theta2 phi1 phi2`, `CONE`/`CONS`, `TRAP`, `TRD1`, `TRD2`,
`GTRA`, `PCON`, `PGON`, plus boolean `Union`/`Subtraction`/`Intersection`. Confirm
parameter order in `src/geomega/src/MDShape*.cxx`.

Detector types (keyword = class): `MDStrip2D`, `MDStrip3D`, `MDStrip3DDirectional`,
`MDCalorimeter`, `MDVoxel3D`, `MDAngerCamera`, `MDDriftChamber`, `MDScintillator`,
`MDACS` (anticoincidence shield). Common detector properties: `.SensitiveVolume`,
`.StructuralOffset`, `.Offset`, `.StripNumber`/`.VoxelNumber`, `.NoiseThreshold`,
`.TriggerThreshold`, `.EnergyResolution <Gaus|Lorentz|...> Ein Epeak sigma` (one line per
calibration energy), `.DepthResolution E sigma`, `.FailureRate`, `.Overflow`,
`.GuardringEnergyResolution`. **Exact properties vary per detector type — read the
matching `src/geomega/inc/MD<Type>.h`.**

Materials (typically in an included `Materials.geo`): `Material <Name>` then
`<Name>.Density`, `<Name>.Component <Z|element> <atoms|fraction> ...`,
`<Name>.ComponentByAtoms` / `.ComponentByMass`.

---

## 2. `.source` — cosima simulation input

Verified beginner example (`resource/examples/introduction/CrabOnly.source`):

```text
Version         1
Geometry        ${MEGALIB}/resource/examples/geomega/cosiballoon/COSIBalloon.12Detector.geo.setup

PhysicsListEM   LivermorePol            # EM physics; LivermorePol enables polarized Compton
StoreSimulationInfo all                 # store full truth in the .sim

Run SpaceSim                            # declare a run named SpaceSim
SpaceSim.FileName   CrabOnlySimulation  # output base name -> CrabOnlySimulation.inc1.id1.sim
SpaceSim.NTriggers  50000               # stop after 50000 triggered events (or use .Time)

SpaceSim.Source Crab                    # declare a source named Crab inside the run
Crab.ParticleType  1                    # 1 = gamma
Crab.Beam          FarFieldPointSource 20 -20   # theta=20 phi=-20 (deg), source at infinity
Crab.Spectrum      PowerLaw 100 10000 2.17       # Emin Emax photonIndex (keV)
Crab.Flux          0.0565                          # ph/cm^2/s
```

Grammar overview:
- **Global**: `Version`, `Geometry <file>`, `Include`, `PhysicsListEM <None|Standard|
  Livermore|LivermorePol|LivermoreG4LECS|Penelope>`, `PhysicsListHD <None|QGSP-BIC-HP|
  QGSP-BERT-HP|FTFP-BERT-HP|FTFP-INCLXX-HP|QGSP-INCLXX-HP>`, `StoreSimulationInfo
  <all|init-only|ia-only|none>`, `StoreCalibrated`, `StoreOneHitPerEvent`,
  `DiscretizeHits`, `DefaultRangeCut`, `DetectorTimeConstant`, `CheckForOverlaps res tol`,
  `BlackAbsorbers`, `DecayMode <...>`.
- **Run block**: `Run <Name>` then `<Name>.FileName`, **either** `<Name>.NTriggers <N>`
  **or** `<Name>.Time <seconds>`, `<Name>.Triggers`, `<Name>.IsotopeProductionFile`,
  `<Name>.OrientationSky`/`.OrientationDetector` (pointings), and one or more
  `<Name>.Source <SourceName>`.
- **Source block**: `<Src>.ParticleType <id>` — MEGAlib particle IDs:
  **1=gamma, 2=e⁺ (positron), 3=e⁻ (electron), 4=proton, 6=neutron** (source-verified in
  `MCSource.cc`; the Cosima.pdf table is wrong for IDs ≥4). **Authoritative full ID table:
  `references/sim-format.md`.** Then `<Src>.Beam <type> <args>`,
  `<Src>.Spectrum <type> <args>`, `<Src>.Flux <ph/cm²/s>`, `<Src>.Polarization <...>`,
  `<Src>.Position`, `<Src>.Orientation`, `<Src>.Successor`/`.IsSuccessor` (cascades),
  `<Src>.LightCurve`.
- **Beam types** (far-field at infinity vs near-field local): `FarFieldPointSource theta phi`,
  `FarFieldAreaSource thMin thMax phiMin phiMax`, `FarFieldGaussian`, `FarFieldIsotropic`,
  `FarFieldFileZenithDependent`, `PointSource x y z`, `HomogeneousBeam x y z dx dy dz radius`,
  `RadialProfileBeam`, `MapProfileBeam`, `Disk`, `Cone`, `IlluminatedDisk`,
  `ActivationVolume`, `Volume`. Arg order is type-specific — see `MCSource::SetBeamType`.
- **Spectrum types**: `Mono E`, `Linear`, `PowerLaw Emin Emax index`,
  `BrokenPowerLaw Emin Ebreak Emax i1 i2`, `CutOffPowerLaw Emin Emax index Ecut`,
  `Comptonized Emin Emax index Epeak`, `Gaussian Emean sigma`, `ThermalBremsstrahlung`,
  `BlackBody Emin Emax T`, `BandFunction Emin Emax alpha beta Epeak`,
  `FileDifferentialFlux <file>`, `NormalizedFunctionInPhPerCm2PerSPerKeV`.
- **Polarization**: `None`, `Random`, `Absolute px py pz degree`, and axis-relative modes.
- **Activation** (two/three-step radioactivation): `Activator <Name>` blocks plus
  `DecayMode ActivationBuildup`/`ActivationDelayedDecay`; see
  `resource/examples/unittests/activations/ActivationStep{1,2,3}.source` and `cosima-simulation.md`.

---

## 3. `.sim` — simulated events

> **For this repo, `references/sim-format.md` is authoritative** — it has the source-verified
> 24-field IA line map, HTsim spec, ANNI ground-truth semantics, and the exact `data/Activation.sim`
> (Version 101) layout. Use that for any real parsing; the overview below is the general shape.

ASCII (or `.bin`) stream: a header, then per-event records. Conceptual structure:

```text
# header
Version 25
Type SIM
Geometry /path/to.geo.setup
...
TB <time>                          # start observation time
# repeated per event:
SE                                 # Start Event delimiter
ID <triggerID> <simEventID>
TI <time>                          # event time (s)
IA <type> <id> <parentID> <detID> <t> <x y z> <dirx diry dirz> <polx poly polz> <E> ...
   #   ^ INIT/COMP/PHOT/PAIR/BREM/RAYL/... interaction truth lines
HTsim <detID>; <x y z>; <E>; <t>; <origins...>    # simulated hit (truth, before noising)
...
EN <energy>                        # (footer-ish counters appear at end of file)
TS <nTriggeredEvents>
```

Key line keywords: `SE` (start event), `ID`, `TI`, `IA` (interaction — the first `IA` of
type `INIT` is the primary), `HTsim`/`HT` (hits), `ED` (energy deposit), `EC` (escaped),
`NS`/`PM` (passive material), `GR` (guard-ring), `BD` (bad flag), `CC` (comment), `XE`/`SE`
boundaries. The `IA` interaction type strings (`INIT`, `COMP`, `PHOT`, `PAIR`, `RAYL`,
`BREM`, `ANNI`, `ESCP`, `ENTR`, `EXIT`, `INEL`, `CAPT`, `DECA`…) encode the physics.
**Confirm the exact column layout in `src/sivan/src/MSimEvent.cxx`,
`MSimIA.cxx`, `MSimHT.cxx`** — column counts depend on `StoreSimulationInfo` level and
output version.

---

## 4. `.tra` — reconstructed (tracked) physical events

> **For this repo, `references/revan-tra.md` is authoritative** for the `.tra` format, the
> per-event-type lines (`CE`/`CD`/`PE`/`PP`/…), and the sim↔tra ID-join gotcha. Use that;
> the table below is the general event-type map.

Header (`Type tra`, `Version`, `Geometry`, …) then per-event blocks. Each event begins
`SE` and carries an event-type (`ET`) tag and reconstructed quantities. Event-type tags
(source-verified) map to the `MPhysicalEvent` subclasses:

| Tag | Event class | Meaning |
|-----|-------------|---------|
| `ET CO` | `MComptonEvent` | Compton event |
| `ET PA` | `MPairEvent` | pair event |
| `ET PH` | `MPhotoEvent` | photo-absorption event |
| `ET PT` | `MPETEvent` | PET coincidence |
| `ET MU` | `MMuonEvent` | muon |
| `ET DY` | `MDecayEvent` | decay |
| `ET MT` | `MMultiEvent` | multi-event |
| `ET UN` | `MUnidentifiableEvent` | unidentified |

Common lines: `SE`, `ID`, `TI <time>`, `CE <E1> <dE1> <E2> <dE2>` (Compton scattered &
recoil energies), `CD <x y z dx dy dz ...>` (interaction positions), `TR` (track dir),
`SQ` (sequence), `GX` (galactic pointing), `RX/RZ` (orientation), `BD` (bad). Reconstructed
geometry uses the first two interaction positions to form the Compton scatter direction.
**Exact line set lives in `MComptonEvent::ParseLine` / `MPhysicalEvent::ParseLine`** under
`src/global/misc/src/`. This is the file MIMREC images.

---

## 5. `.evta` — calibrated measured hits (input to revan)

The measured-data analogue of `.sim` but only calibrated hits (no truth). Header
(`Type EVTA`, `Version`, `Geometry`) then per-event `SE`/`ID`/`TI` plus `HT` hit lines
(`HT detID; x y z; E; t`). Produced by the detector-effects engine / fretalon from real
data; consumed by revan to produce `.tra`. Parser: `MFileEventsEvta.cxx`.

---

## 6. `.roa` — raw read-outs (input to fretalon/melinator)

Raw, uncalibrated channel read-outs (ADC counts, optionally timing/temperature). Lines
declare a read-out element (e.g. strip detector/side/strip) and its data. Parser:
`src/fretalon/base/src/MFileReadOuts.cxx`; data model in `MReadOut`, `MReadOutElement*`,
`MReadOutData*`. Used to derive `.ecal` energy calibrations.

---

## 7. `.rsp` — response matrices

N-dimensional histogram serialization (optionally gzip). Text header with metadata
(`Type ResponseMatrixO<N>` or `...ON`, `NM` name, `OD` order/dimension, `TS` total
simulated events, `SA` start-area radius, `CE` centered flag, `SP` sparse flag, `HA` hash),
then per-axis definitions (`AN` axis name, `AT` axis type — `1D BinEdges` or `2D FISBEL`
for spherical, `AD` axis data/bin-edges), then the data either as a dense
`StartStream…StopStream` block or sparse `RD <indices> <value>` lines. Read/written by
`MResponseMatrixO1..O17`/`MResponseMatrixON` (+ `MResponseMatrixAxis*`) in
`src/global/misc/src/`. Inspect/manipulate with the `mresponsemanipulator` tool
(`MResponseManipulator`: statistics, append, collapse, divide, ratio, probability).

---

## 8. `.cfg` — XML application configuration

Plain XML (`MSettings*` / `MXmlDocument`). One schema per app: `RevanConfigurationFile`,
`MimrecConfigurationFile`, etc. Holds geometry path, data-file history, algorithm
selections, thresholds (`<Min>/<Max>` pairs), response-file paths, event-selection cuts.
Edit via the GUI, or override single values on the command line with `-C
Path.To.Key=Value` (see `revan-reconstruction.md` / `mimrec-imaging.md`). Examples:
`resource/examples/introduction/CrabOnly.revan.cfg`, `CrabOnly.mimrec.cfg`.

> Because these are generated/consumed by the GUIs, the most reliable way to get a valid
> `.cfg` is to open the tool's GUI once, set options, and save — then hand-edit or script
> `-C` overrides. Hand-authoring from scratch is error-prone.
