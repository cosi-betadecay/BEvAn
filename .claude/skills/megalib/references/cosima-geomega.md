# cosima & geomega essentials

Light-touch reference: enough to read the geometry this data was simulated
on and to author a controlled simulation (e.g. a clean β⁺ point source for
validating the annihilation-axis reconstruction).

## cosima invocation

```
cosima [-s <seed>] [-u] [-v <0-3>] MyRun.source
```
`-s` random seed (parallel runs need distinct seeds; `mcosima` automates
fan-out), `-u` = don't gzip the output .sim.

## .source file format

Global section: `Version 1`, `Geometry <path>`, `PhysicsListEM LivermorePol`,
`PhysicsListHD qgsp-bic-hp` (needed for activation), `DecayMode`,
`StoreSimulationInfo all`, `DefaultRangeCut 0.1`.

Run + source sections (real example values from
`resource/examples/cosima/source/*.source`):

```
Run MyRun
MyRun.FileName     MyOutput
MyRun.NTriggers    100000        # or MyRun.Time <seconds>

MyRun.Source S1
S1.ParticleType    1             # particle IDs: 1=γ, 2=e+, 3=e- (same scheme as sim files)
S1.Beam            PointSource 0 0 0          # isotropic near-field point at (x,y,z) cm
S1.Spectrum        Mono 511                   # keV
S1.Flux            1.0
```

Beam types seen in the examples: `PointSource x y z` (near-field
isotropic), `FarFieldPointSource theta phi`, `FarFieldAreaSource`,
`BoxSource x y z dx dy dz`, `FarFieldGaussian`, file-based beams.
Spectrum types: `Mono E`, `Line`, `PowerLaw`, `BrokenPowerLaw`,
`BlackBody`, `File <dat>`.

A genuine β⁺ validation source: `ParticleType 2` (positron) with a low
`Mono` energy at a fixed point — the e⁺ stops and annihilates, giving
controlled ANNI pairs. (For pure back-to-back photons without β physics
you'd need two correlated γ sources, which cosima does not couple — use
the positron.)

### Activation simulations (how Activation.sim was made)

Three-step workflow (`resource/examples/cosima/source/ActivationStep*.source`):
1. **Step 1** — irradiate: simulate primaries with `DecayMode
   ActivationBuildup`, store produced isotopes (`Isotopes...dat`).
2. **Step 2** — `ActivationMode` calculation: convert isotope counts +
   irradiation/cooldown times into an activation inventory
   (`Activation.dat`).
3. **Step 3** — decay: `DecayMode ActivationDelayedDecay`,
   `Run.ActivationSources Activation.dat`, `Run.Time <seconds>` (must be
   time-limited, not trigger-limited). Output = our kind of file:
   `BeamType Activation`, decays of activated nuclei distributed through
   the detector materials.

## The geometry: Max.geo.setup (resource/examples/geomega/special/)

Simplified COSI-like germanium stack used to simulate this repo's data:
- World: 5000 cm vacuum box.
- 4 identical layers of **Strip3D germanium wafers** stacked in z, ~8x8 cm²
  active area each, strip readout (~40x40), guard rings vetoing edge events,
  aluminum cryostat around each layer.
- Gaussian energy resolution, sub-keV to ~keV σ over 10-1000 keV.
- Trigger: any active-Ge hit; guard-ring hits veto.
- Hence all HTsim lines in Activation.sim carry **detector type 3 = Strip3D**.

## geomega .geo.setup format (one paragraph)

Volumes (shape BRIK/TUBE/SPHE..., material, position, mother) form a
hierarchy under a world volume; `Include` pulls in shared files (materials
library); `Constant` defines parameters; detector blocks (`Strip3D`,
`Strip2D`, `Calorimeter`, `Scintillator`, ...) declare a sensitive volume,
its segmentation (strips/voxels), energy/position/time resolutions,
thresholds; `Trigger` blocks define the trigger/veto logic.

## Detector type IDs (MDDetector.cxx:60-69 — first field of HTsim lines)

| ID | Type | ID | Type |
|---|---|---|---|
| 1 | Strip2D | 6 | Strip3DDirectional |
| 2 | Calorimeter | 7 | AngerCamera |
| 3 | **Strip3D** (this data) | 8 | Voxel3D |
| 4 | Scintillator / ACS | 9 | GuardRing |
| 5 | DriftChamber | | |
