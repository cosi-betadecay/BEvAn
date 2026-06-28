# COSIMA — Geant4 Monte-Carlo simulation

Program: `cosima`. Library: `libCosima` (links Geant4). Classes prefixed `MC`. Source:
`src/cosima/{inc,src,unittests}` — note cosima uses Geant4-style **`.hh`/`.cc`** extensions.
Input: a `.source` file (`file-formats.md §2`). Output: `.sim`(.gz/.bin).

## What it does

Runs Geant4 particle transport through a Geomega geometry and writes every interaction +
the resulting detector hits (with truth) to a `.sim` file. The `.source` file defines the
geometry, physics, one or more **runs**, and within each run one or more **sources**
(particle type + beam geometry + spectrum + flux + polarization).

## Architecture & data flow

```
MCMain (CLI parse) → MCRunManager (G4RunManager)
   ├ MCParameterFile        parse .source into MCRun + MCSource list
   ├ MCDetectorConstruction Geomega geometry → Geant4 volumes, regions, cuts
   ├ MCPhysicsList          select EM + hadronic physics
   ├ MCPrimaryGeneratorAction   MCSource → primary particles (G4GeneralParticleSource)
   ├ MCSteppingAction/MCTrackingAction   per-step/track truth bookkeeping
   └ MCEventAction          apply triggering, build MSimEvent, write .sim (or stream TCP/IP)
```

Sensitive-detector / hit classes mirror the Geomega detector types: `MC2DStripSD`,
`MCCalorBarSD`, `MCScintillatorSD`, `MCVoxel3DSD`, `MCAngerCameraSD`, `MCDriftChamberSD`
(+ matching `*Hit`), base `MCSD`/`MCVHit`.

## Physics (`MCPhysicsList`)

- **EM** (`PhysicsListEM`): `Standard`, `Livermore` (low-energy, high-Z),
  **`LivermorePol`** (adds **polarized Compton** — needed for polarization studies),
  `LivermoreG4LECS`, `Penelope`.
- **Hadronic** (`PhysicsListHD`, default `None`): `QGSP-BIC-HP`, `QGSP-BERT-HP`,
  `FTFP-BERT-HP`, `FTFP-INCLXX-HP`, `QGSP-INCLXX-HP` (needed for cosmic-ray backgrounds,
  activation, hadron therapy). `-HP` = high-precision neutron transport.

## Sources, beams, spectra (`MCSource`)

`MCSource` is the big, flexible class. A source = particle type + beam + spectrum + flux
(+ polarization, light curve, orientation). See the keyword lists in `file-formats.md §2`.
Highlights:
- **Far-field** (astrophysical, source at infinity, specified by θ,φ): point, area, Gaussian,
  isotropic, zenith-dependent-from-file, disk.
- **Near-field** (local, x/y/z): point, homogeneous beam, disk, cone, fan, radial/2D-map
  profile, illuminated disk/square, activation volume — for lab/medical/calibration setups.
- **Spectra**: Mono, PowerLaw, Broken/CutOff power law, Comptonized, Gaussian,
  ThermalBremsstrahlung, BlackBody, **BandFunction** (GRBs), file-based differential flux.
- **Polarization**: none/random/absolute-vector/axis-relative with a degree 0–1.
- **Light curves & cascades**: time-variable flux from file; `Successor` chains model
  decay cascades / correlated emission.

## Activation (radioactivation) — `MCActivator`, `MCIsotopeStore`

Two/three-step workflow to model material becoming radioactive in orbit and then decaying:
1. **Step 1** — irradiate the geometry (e.g. with cosmic protons/neutrons) and record
   isotope production (`IsotopeProductionFile`).
2. **Step 2** — `MCActivator` builds up activity over a mission time (constant irradiation,
   with/without cooldown, or a time profile), producing an isotope store / source list.
3. **Step 3** — simulate the **delayed decays** as a radioactive source (instrument
   background).
Modes via `DecayMode` (`ActivationBuildup`, `ActivationDelayedDecay`, etc.) and `Activator`
blocks. Analytic decay-chain solutions up to 5 isotopes plus numerical/MC options. Worked
example: `resource/examples/unittests/activations/ActivationStep{1,2,3}.source`, and
`resource/examples/advanced/{ActivationPlotter,CoolDown,TGRS}`.

## Running cosima

```bash
cosima Run.source                 # single-threaded
cosima -s 12345 Run.source        # fixed random seed (reproducible)
cosima -v 2 Run.source            # higher verbosity
cosima -u Run.source              # do NOT gzip the .sim (default is gzip)
```

Output naming: `<FileName>.inc<incarnation>.id<parallelID>.sim[.gz]`. (Confirm exact flags
in `Usage()` of `src/cosima/src/MCMain.cc`: typically `-s` seed, `-v` verbosity, `-u`
no-gzip, `-z` gzip, `-p`/`-f` parallel/incarnation IDs used by wrappers, `-i` interactive,
`-m` Geant4 macro.)

### Parallel / distributed

- **`mcosima -t <Nruns> -m <maxParallel> [-w] Run.source`** — launch many cosima instances
  on the local machine (one per core), then it writes a concatenation `.p1.sim` master.
  `-w` waits for completion; `-z`/`-u` control gzip.
- **`mpicosima`** — MPI variant across an HPC cluster (uses the `bundler` program).
- **`dcosima -s=Run.source -n=<name> -i=<instances>`** — distribute across SSH-reachable
  machines listed in `~/.dmegalib.cfg`; companions `dcosima-assemblefiles`, `-rsync`,
  `-listrunning`, `-kill`. See `tools-scripts.md`.
- Merge per-instance outputs with **`msimconcatter *.sim`** → a single `.sim` (or use the
  generated `.p1.sim` concatenation file that references the parts).

## Gotchas

- Always check `CheckForOverlaps` output / run geomega overlap check first — overlaps
  silently corrupt transport.
- `StoreSimulationInfo all` is needed for response creation and sivan truth analysis;
  `init-only`/`none` shrink files but drop truth.
- Geant4 dataset env vars (`G4LEDATA`, `G4LEVELGAMMADATA`, etc.) must be set (the source
  script does this) or physics initialization fails.
- Polarization studies require `PhysicsListEM LivermorePol`.
- `NTriggers` counts *triggered* events (post-trigger); `Time` simulates a wall-clock
  exposure given the flux. Use one or the other per run.
