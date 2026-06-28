# Tools, scripts, addons & language bindings

`bin/` holds the built program symlinks plus ~40 wrapper/utility shell scripts.
`src/addon/` holds ~25 standalone C++ programs. Plus MEGAlib is scriptable from
Python/Julia via PyROOT.

## bin/ — categorized

### Environment / config
- `source-megalib.sh` (or generated `env.sh`) — sets `$MEGALIB`, paths. Source it first.
- `megalib-config` — query versions: `--root`, `--geant4`, `--compiler`, `--python3`.
- `mlink`, `generatelinkdef` — build-time helpers (ROOT dictionary / symlinks).

### Local multi-core wrappers (one machine, many cores)
- `mcosima -t <runs> -m <maxParallel> [-w] [-z|-u] Run.source` — parallel cosima.
- `mrevan -c cfg -t <threads> [-g geo] [-o] -f *.sim` — parallel revan (`-f` last).
- `mresponsecreator …` — parallel responsecreator over multiple sim files.
- `mpicosima` — MPI cosima across a cluster (uses the `bundler` program).

### Distributed across SSH machines (`~/.dmegalib.cfg` defines the pool)
- `dcosima -s=Run.source -n=<name> -i=<instances> [-z]` — distributed simulation.
  Helpers: `dcosima-assemblefiles`, `dcosima-rsync`, `dcosima-listrunning`,
  `dcosima-kill`, `dcosima-killall`, `dcosima-clean`, `dcosima-runinstance`, `dcosima2`.
- `drevan --datadir=… --targetdir=… -g=… -c=…` — distributed reconstruction.
- `dresponsecreator --datadir=… --targetdir=… -g=… -r=… -m=… -c=… -b=…` — distributed responses.
- Cluster management `dmegalib-*`: `dmegalib-setup`, `dmegalib-checkmegalib`,
  `dmegalib-checkaccessibility`, `dmegalib-allowedinstances`, `dmegalib-listallinstances`,
  `dmegalib-updatemegalib`, `dmegalib-responsetime`, `dmegalib-showdiskspace`,
  `dmegalib-gettasks`, `dmegalib-runninginstances`, `dmegalib-killall`.

### File utilities
- `msimconcatter *.sim` → merge simulation files; `mtraconcatter *.tra` → merge tracked files.
- `mzip`, `mpack`, `msync`, `mreplace`, `mpopulate`, `repair`.
- `mwait`, `mwaitforload`, `mdelay`, `mtime`, `mmaxmem`, `mlines`, `mvalgrind`, `debug`,
  `launch` — job/timing/profiling helpers.

> All wrapper scripts: run with `-h`/no args to see usage, or read the script directly in
> `bin/` (they're short bash). Flag conventions differ between the `m*` (space-separated,
> `-t 8`) and `d*` (equals-form, `--instances=8`) families.

## src/addon/ — standalone programs

Built into `bin/` (some only when HEASoft/cfitsio is present). Notable ones:

| Program | Purpose |
|---------|---------|
| `SensitivityOptimizer` | optimize event selections for best sensitivity (large tool) |
| `SimCombiner` / `msimconcatter` | merge `.sim` files |
| `SimRewriter` / `SimBinaryConverter` | rewrite / convert `.sim` formats |
| `SimRandomCoincidence` | inject random coincidences into simulations |
| `TraMerger` / `mtraconcatter` | merge `.tra` files |
| `TraAnalyzer` / `SimAnalyzer` | quick analysis of tracked/sim files |
| `DecayAnalyzer` | radioactive-decay analysis |
| `SimpleComptonImaging` | minimal Compton imaging example/tool |
| `Revoxelizer`, `Discretizer` | re-voxelize / discretize detector responses |
| `BackgroundMixer` | combine background simulations |
| `ResponseToXSPEC` | export response → XSPEC format (needs cfitsio) |
| `TraFitsConverter` | `.tra` → FITS (needs cfitsio) |
| `ConvertMGGPOD`, `ConvertMGeant`, `ConvertACTtools` | format converters |
| `Mask`, `IsotopeFileSplitter`, `ShowHistograms`, `CompareHistograms` | misc helpers |
| `bundler` | MPI job distributor used by `mpicosima` |

`eview` — the standalone **event viewer** (`src/eview/`): visualize hits/events of a
`.sim`/`.tra` in the geometry.

## Python / Julia bindings (PyROOT)

MEGAlib has no separate Python package — it is used through **ROOT's PyROOT** reflection of
the C++ classes (enabled by the ROOT dictionaries). Examples:
`resource/examples/python/` and `resource/examples/julia/`.

Python pattern:
```python
import ROOT as M
M.gSystem.Load("libMEGAlib")          # or the full path to lib/libMEGAlib.so
G = M.MGlobal(); G.Initialize()

Geo = M.MDGeometryQuest()
Geo.ScanSetupFile(M.MString("Detector.geo.setup"))

Reader = M.MFileEventsSim(Geo)
Reader.Open(M.MString("Sim.inc1.id1.sim.gz"))
while True:
    Event = Reader.GetNextEvent()
    if not Event:
        break
    M.SetOwnership(Event, True)        # IMPORTANT: you own it, prevents a leak
    for i in range(Event.GetNHTs()):
        HT = Event.GetHTAt(i)
        print(HT.GetEnergy(), HT.GetPosition().X())
Reader.Close()
```
Julia is the same via `PyCall` (`M = pyimport("ROOT")`); build PyCall against the same
`python3` MEGAlib uses (`ENV["PYTHON"]=...; Pkg.build("PyCall")`).

Example scripts to crib from: `AnalyzeSimFile.py`, `AnalyzeEvents.py`, `FitSpectrum.py`,
`ARMFitter.py`, `KleinNishinaAverageScatterAngle.py`, `GeometryTricks.py`,
`TraRewriter.py`, `MakeLogBins.py`.
