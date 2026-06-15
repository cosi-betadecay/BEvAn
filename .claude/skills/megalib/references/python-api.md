# MEGAlib Python (PyROOT) API — the classes this project uses

MEGAlib's C++ classes are exposed to Python via PyROOT, so the public
headers ARE the Python API. Headers live under
`/Users/aryaraeesi/Documents/COSI/megalib/src/{sivan,geomega,global/misc}/inc/`.

```python
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.MGlobal().Initialize()
```

Python gotchas: `MString` needs `.Data()` or `str()` (or compare with
`== M.MString("ANNI")`); `MVector` components via `.X()/.Y()/.Z()`;
`GetIAAt(i)`/`GetHTAt(i)` are 0-based but IA IDs/OriginIDs are 1-based.

## MFileEventsSim (src/sivan/inc/MFileEventsSim.h)

- `MFileEventsSim(MDGeometryQuest* geo)`; `Open(MString file)` → bool
- `GetNextEvent(bool Analyze=True)` → `MSimEvent*`, None/NULL at EOF.
  Loop: `iter(lambda: reader.GetNextEvent(), None)`. Wrap each event in
  `M.SetOwnership(event, True)` to avoid leaks.
- `GetSimulatedEvents()` → started-event count (valid after full read).

## MSimEvent (src/sivan/inc/MSimEvent.h)

- IDs/time: `GetID()`, `GetSimulationEventID()`, `GetTime()` (MTime)
- IAs: `GetNIAs()`, `GetIAAt(i)`, `GetIAById(id)` (by 1-based ID)
- Hits: `GetNHTs()`, `GetHTAt(i)`, `GetAllHTOrigins()` → vector<int>
- Truth↔hit linking helpers (useful for the ANNI work):
  - `GetDescendents(iaId)` → all IA IDs descending from iaId (incl. itself)
  - `GetHTListIncludingDescendents(iaId)` → all hits caused by that IA's tree
  - `IsIACompletelyAbsorbed(iaId, absTol, relTol)`
  - `IsIAResolved(iaId)` — hits contain only this IA's descendants
- Energy: `GetTotalEnergyDeposit()` (after noising),
  `GetTotalEnergyDepositBeforeNoising()`, `IsCompletelyAbsorbed(tol)`
- Clusters: `GetNClusters()`, `GetClusterAt(i)`, `CreateClusters()`
- `GetPhysicalEvent(Ideal=False)` → MPhysicalEvent (revan-style view)

## MSimIA (src/sivan/inc/MSimIA.h)

- `GetProcess()` → MString ("ANNI", "COMP", ...); `GetID()`,
  `GetOriginID()`, `GetDetectorType()`, `GetTime()`
- `GetPosition()` → MVector [cm]
- Mother (incoming particle, post-interaction): `GetMotherParticleID()`,
  `GetMotherDirection()`, `GetMotherPolarisation()`, `GetMotherEnergy()`
- Secondary (created particle): `GetSecondaryParticleID()`,
  `GetSecondaryDirection()` (→ for ANNI: the photon emission unit vector),
  `GetSecondaryPolarisation()`, `GetSecondaryEnergy()`
- Deprecated aliases exist (`GetType`→`GetProcess`, `GetOrigin`→
  `GetOriginID`, `GetDetector`→`GetDetectorType`, bare `GetDirection`/
  `GetEnergy` = secondary) — prefer the explicit names.

## MSimHT (src/sivan/inc/MSimHT.h)

- `GetDetectorType()`, `GetPosition()` → MVector, `GetEnergy()`,
  `GetTime()` (all post-noising; `GetOriginalPosition/Energy/Time()` for
  pre-noising values)
- Origins: `GetNOrigins()`, `GetOriginAt(i)`, `GetOrigins()` → vector<int>,
  `IsOrigin(iaId)` → bool, `GetSmallestOrigin()`
- `GetCluster()` / cluster association

## MVector (src/global/misc/inc/MVector.h)

- `X() Y() Z()`, `Mag()`, `Mag2()`, `Theta()`, `Phi()`
- `Unit()` (normalized copy), `Unitize()` (in-place), `IsNull()`
- `Dot(v)` → double, `Cross(v)` → MVector, `Angle(v)` → radians
- `DistanceToLine(A, B)` → point-to-line distance (handy for
  line-of-response vs vertex checks)
- operators `+ - *` (scalar), unary `-`; `AreEqual(v, tol)`

## MDGeometryQuest (src/geomega/inc/MDGeometryQuest.h)

- `ScanSetupFile(MString file)` → bool (load before opening sim files)
- `GetVolume(pos)`, `GetMaterial(pos)`, `IsSensitive(pos)`,
  `GetDetectorName(pos)`
- `Noise(pos, energy, time)` — apply detector response;
  `GetResolutions(...)`
- Absorption probabilities along a path:
  `GetAbsorptionProbability(start, stop, E)` and per-process variants
  (Photo/Compton/Pair/Rayleigh) — useful for escape-probability reasoning.

## MFileEventsTra / MPhysicalEvent / MComptonEvent (src/global/misc/inc/)

- `MFileEventsTra().Open(file)`; `GetNextEvent()` → MPhysicalEvent*
- MPhysicalEvent: `GetType()` (c_Compton/c_Photo/c_Pair/...),
  `GetTypeString()`, `GetId()` (= originating sim event ID), `GetTime()`,
  `GetEnergy()`/`Ei()`, `GetPosition()`
- MComptonEvent (the useful getters): `Eg()`/`Ee()` (scattered γ / recoil e⁻
  energies), `C1()`/`C2()` (first/second interaction positions), `De()`
  (electron direction), `Phi()` (Compton scatter angle), `Di()`/`Dg()`
  (initial/scattered γ directions), `SequenceLength()`, `LeverArm()`,
  `ComptonQualityFactor1()`, `GetARMGamma(testPos)`; statics:
  `ComputeEeViaThetaEg`, `GetKleinNishina(Ei, phi)`.
