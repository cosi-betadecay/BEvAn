# REVAN — event reconstruction

Program: `revan` (Real EVent ANalysis). Library: `libRevan`. Classes prefixed `MER`
(algorithms) and `MRE` (data). Source: `src/revan/{inc,src}`. Turns raw hits
(`.sim` from cosima, `.evta` from measured data) into reconstructed **physical events**
(`.tra`) — i.e. decides "this was a Compton event scattering here then absorbed there".

> **For this repo, `references/revan-tra.md` is the authoritative companion** — it has the
> source-verified revan CLI flags, the `.tra` event-type codes (`CO`/`PH`/`PA`/`MU`/`DY`/
> `PT`/`MT`/`UN`), the per-event-type line fields, and the critical **sim↔tra ID-join**
> gotcha (tra IDs are a gappy subset — join on `ID`, never positionally). This file is the
> broader algorithm/pipeline view.

## Pipeline

```
raw hits (MRERawEvent of MREHit)
  → noising / calibration            MERNoising
  → coincidence search (time window) MERCoincidence
  → hit clustering                   MERHitClusterizer
  → event clustering (split γ's)     MEREventClusterizer
  → electron tracking                MERTrack*
  → Compton sequence reconstruction  MERCSR*
  → MPhysicalEvent (Compton/Pair/Photo/...) → .tra
```

Orchestrated by **`MRawEventAnalyzer`** (`PreAnalysis()` → `AnalyzeEvent()` loop →
`PostAnalysis()`), which maintains competing reconstruction **incarnations** (hypotheses)
ranked by a quality factor. Data elements derive from **`MRESE`** (Raw Event Sequence
Element): `MREHit`, `MRECluster`, `MRETrack`, `MRERawEvent`.

## Stage algorithms (each selectable in the `.cfg`)

- **Hit clustering** (`MERHitClusterizer`): by distance (per-detector thresholds), by
  adjacent voxels (levels), or by PDF/response.
- **Event clustering** (`MEREventClusterizer{Distance,TMVA,DataSet}`): separate
  multiple incident gammas in one frame.
- **Electron tracking** (`MERTrack` + `ChiSquare`, `Bayesian`, `Pearson`, `Gas`,
  `Directional`, `Rank`): find e⁻ tracks, pair vertices, MIPs.
- **Compton Sequence Reconstruction / CSR** (`MERCSR` + variants):
  - `MERCSRChiSquare` — Chi²/test-statistic ordering (most common, robust).
  - `MERCSRBayesian` — uses a Bayesian response matrix (`.rsp`) → needs a Compton response
    built by `responsecreator`.
  - `MERCSRTMVA` / `MERCSRNeuralNetwork` — machine-learning sequencing (TMVA-trained).
  - `MERCSRToF` / `MERCSRToFWithEnergyRecovery` — time-of-flight based.
  - `MERCSREnergyRecovery` — recovers escaped energy.
- **Coincidence** (`MERCoincidence`): merge hits within a time window (for measured data).
- **Decay** (`MERDecay`): flag radioactive-decay events.

Configuration container: `MSettingsEventReconstruction` ↔ the `.revan.cfg` XML. Algorithm
choices, distance thresholds, energy/lever-arm windows, response-file paths, and quality
cuts all live there.

## Running revan

The `.revan.cfg` typically already references the geometry and the data file (set them via
the GUI once). Common forms (confirm flags in `Usage()` of
`src/revan/src/MRevanMain.cxx`):

```bash
# GUI
revan -c CrabOnly.revan.cfg

# Batch reconstruct a .sim/.evta into a .tra (no GUI)
revan -g Detector.geo.setup -c CrabOnly.revan.cfg -f Sim.inc1.id1.sim -n -a

# Override a single config value without editing the file (repeatable)
revan -c X.revan.cfg -f data.sim -n -a -C CoincidenceWindow=1e-6

# Save origin info (needed when this .tra feeds responsecreator)
revan -c X.revan.cfg -f data.sim -n -a --oi
```

Key options (verify): `-g` geometry, `-c` config, `-f` input file, `-a`/`--analyze` batch
analyze, `-n`/`--no-gui` headless, `-o` output name, `-C key=value` config override
(repeatable, dotted path), `--oi` store origin/IA info, `-d` debug, `-s` generate spectra.
Output: `<input>.tra[.gz]`.

### Parallel / distributed

- **`mrevan -c cfg -t <threads> -f *.sim`** — reconstruct many files in parallel on one
  machine (`-f` must be last; `-g` overrides geometry, `-o` saves origins).
- **`drevan --datadir=… --targetdir=… -g=… -c=…`** — distribute across the cluster.
- Concatenate results with **`mtraconcatter *.tra`**.

## Tuning / debugging notes

- Reconstruction quality is dominated by the CSR choice. **Chi²** needs no response file and
  is the safe default; **Bayesian/TMVA** can do better but require a matching response built
  from the *same geometry* via `responsecreator` (modes `cb`/`ct`, see `sivan-response.md`).
- Energy/lever-arm/quality windows in the `.cfg` strongly shape which events survive into
  the `.tra`; loosening them increases statistics but adds mis-reconstructed events.
- `--oi`/origin info must be enabled when the `.tra`/intermediate is used to build responses
  or to compare against truth.
- To validate reconstruction against truth, pair with **sivan** on the same `.sim`
  (`sivan-response.md`).
- Inspect resulting events visually with **`eview`** (`tools-scripts.md`).
