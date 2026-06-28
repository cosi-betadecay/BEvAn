# End-to-end workflows (recipes)

Copy-pasteable chains. Always `source …/source-megalib.sh` first so `$MEGALIB` and tools
resolve. Flag spellings: confirm with `<tool> --help` if a command errors (formats/flags
shift across versions). The canonical examples live in `resource/examples/`.

## 1. The canonical tutorial: simulate → reconstruct → image (Crab)

Files: `resource/examples/introduction/CrabOnly.{source,revan.cfg,mimrec.cfg}`.

```bash
cd $MEGALIB/resource/examples/introduction

# 1) Simulate: Crab-like power-law point source through the COSI balloon geometry
cosima CrabOnly.source
#   → CrabOnlySimulation.inc1.id1.sim(.gz)

# 2) Reconstruct raw hits into Compton/physical events
revan -c CrabOnly.revan.cfg -f CrabOnlySimulation.inc1.id1.sim.gz -n -a
#   → CrabOnlySimulation.inc1.id1.tra(.gz)

# 3) Image / analyze the tracked events
mimrec -c CrabOnly.mimrec.cfg -f CrabOnlySimulation.inc1.id1.tra.gz       # GUI: see the image
mimrec -c CrabOnly.mimrec.cfg -f CrabOnlySimulation.inc1.id1.tra.gz -n -a # batch: ARM
mimrec -c CrabOnly.mimrec.cfg -f CrabOnlySimulation.inc1.id1.tra.gz -n -i -o Crab.root  # image
```

The `.revan.cfg`/`.mimrec.cfg` already point at the geometry and define algorithms + event
selections; create/edit them via the GUIs (`revan -c …`, `mimrec -c …`) and Save.

## 2. Faster simulation on all cores, then merge

```bash
mcosima -t 8 -m 8 -w CrabOnly.source     # 8 parallel cosima runs, wait for completion
msimconcatter CrabOnlySimulation*.sim.gz # → one merged .sim (or use the generated .p1.sim)
mrevan -c CrabOnly.revan.cfg -t 8 -f CrabOnlySimulation*.sim.gz   # parallel reconstruct
mtraconcatter CrabOnlySimulation*.tra.gz # merge tracked files
```

## 3. Inspect a geometry / catch overlaps

```bash
geomega -g $MEGALIB/resource/examples/geomega/cosiballoon/COSIBalloon.12Detector.geo.setup
# GUI → Analysis/Geometry: render; check the overlap report in the terminal.
geomega -g My.geo.setup -n --create-cross-sections   # batch: make cross-section files
```

## 4. Build an instrument response, then use it

Imaging response for mimrec:
```bash
# Simulate a representative set with full truth (StoreSimulationInfo all)
cosima ResponseSim.source
responsecreator -g Det.geo.setup -f ResponseSim.inc1.id1.sim.gz \
                -r ImagingResponse -m ib -c My.revan.cfg -b My.mimrec.cfg -z
# Point the .mimrec.cfg response setting at ImagingResponse*.rsp, then run mimrec as usual.
```
Bayesian Compton-sequence response for revan:
```bash
responsecreator -g Det.geo.setup -f ResponseSim.inc1.id1.sim.gz -r Compton -m cb -c My.revan.cfg -z
# In the .revan.cfg select Bayesian CSR and point it at Compton*.rsp, then reconstruct.
```
(See `sivan-response.md` for all `-m` modes. Geometry must match the data.)

## 5. Effective area / sensitivity study

```bash
# resource/examples/{cosima,revan,mimrec} have EffectiveArea.* configs
cosima EffectiveArea.source
revan  -c EffectiveArea.revan.cfg  -f EffectiveArea.inc1.id1.sim.gz -n -a
mimrec -c EffectiveArea.mimrec.cfg -f EffectiveArea.inc1.id1.tra.gz -n -a   # ARM/efficiency
# For optimized event selections: src/addon SensitivityOptimizer.
```

## 6. ARM / FWHM and spectra without the GUI

```bash
mimrec -c X.mimrec.cfg -f data.tra.gz -n -a -o arm.root   # ARM distribution + FWHM
mimrec -c X.mimrec.cfg -f data.tra.gz -n -s -o spec.root  # energy spectrum
mimrec -c X.mimrec.cfg -f data.tra.gz -n -p               # polarization
mimrec -c X.mimrec.cfg -f data.tra.gz -n -x -o selected.tra   # extract selected events
```

## 7. Validate reconstruction against truth

```bash
sivan -g Det.geo.setup -f data.sim.gz          # truth analysis of the same .sim
# Compare sivan truth (energy containment, first-IA detector, absorption) with the
# revan .tra reconstruction quality to assess the algorithm.
```

## 8. Analyze events programmatically (PyROOT)

```python
import ROOT as M
M.gSystem.Load("libMEGAlib"); M.MGlobal().Initialize()
Geo = M.MDGeometryQuest(); Geo.ScanSetupFile(M.MString("Det.geo.setup"))
R = M.MFileEventsTra(Geo); R.Open(M.MString("data.tra.gz"))
n = 0; Etot = 0.0
while True:
    ev = R.GetNextEvent()
    if not ev: break
    M.SetOwnership(ev, True)              # avoid leak
    if ev.GetType() == M.MPhysicalEvent.c_Compton:
        n += 1; Etot += ev.GetEnergy()
R.Close()
print(n, "Compton events, mean E =", Etot/max(n,1), "keV")
```
(For `.sim` use `M.MFileEventsSim` → `MSimEvent` with `GetNHTs()`/`GetHTAt()`.)

## 9. Calibrate real detector data (energy calibration)

```bash
# Raw calibration-isotope read-outs (.roa) → energy calibration (.ecal)
melinator -l            # GUI: load last files, find/fit peaks, determine model, save .ecal
melinator -a -e Cal.ecal  # auto end-to-end
# Then revan/fretalon modules apply the .ecal when processing measured .roa/.evta.
```

## 10. Run a big campaign on a cluster

```bash
# One-time: configure machine pool in ~/.dmegalib.cfg (dmegalib-setup)
dcosima -s=Run.source -n=BigRun -i=50 -z          # 50 distributed instances
dcosima-listrunning                                # monitor
dcosima-assemblefiles ...                          # gather results back
drevan --datadir=/nfs/BigRun --targetdir=/nfs/BigRunTra -g=/nfs/Det.geo.setup -c=/nfs/Run.revan.cfg
```

## Advanced examples worth reading (`resource/examples/advanced/`)
`Background`, `ActivationPlotter`/`CoolDown`/`TGRS` (activation), `BinnedImaging`,
`DetectorEffectsEngine`, `AllSky`, `LightCurves`, `SpectralEvolution`, `InducedPolarization`,
`ComptonPET`/`ComptonSPECT`/`MediMax`/`HadronTherapy` (non-astro applications),
`NeuralNetworks`, `Orientations`, `GRBSampler`, `GradedZShield`, `Atmosphere`, `Pipeline`,
`TriggerMap`, `EventList`. Each is a self-contained, runnable demonstration.
