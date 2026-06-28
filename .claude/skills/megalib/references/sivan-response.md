# SIVAN (truth analysis) & RESPONSE (matrix generation)

Two tightly-related modules: sivan reads simulated truth; response builds the instrument
response matrices by **matching truth (sivan) against reconstruction (revan)**.

---

## SIVAN — simulated-event analysis

Program: `sivan` (SImulated event ANalysis). Library: `libSivan`. Classes prefixed `MSim`.
Source: `src/sivan/{inc,src}`. Reads `.sim` files and analyzes them against ground truth.

### Data model
- **`MSimEvent`** — one simulated event: the full interaction list, hits, primary info.
- **`MSimIA`** — an interaction truth record (process type `INIT/COMP/PHOT/PAIR/…`,
  particle IDs, energy, position, direction, time, parent linkage).
- **`MSimHT`** — a hit truth record (detector, position before/after noising, energy, time,
  list of originating interaction IDs). `Noise()` applies detector response.
- **`MSimGR`** (guard-ring), **`MSimPM`** (passive-material deposit), **`MSimDR`**,
  **`MSimCluster`**.
- Reader: **`MFileEventsSim`** (ASCII `.sim`/`.sim.gz` and binary `.sim.bin`), driven by
  `MSimBinaryOptions` / `StoreSimulationInfo` level. (`MSimEventLoader` is deprecated.)

### What it's for
Because `.sim` carries truth, sivan answers "what really happened?" questions: event-type
classification, which detector saw the first interaction, total deposited vs original
energy, complete vs partial absorption, cluster sizes, efficiency. It is the reference
against which revan's reconstruction quality is measured.

### Running
```bash
sivan -g Detector.geo.setup -f Sim.inc1.id1.sim       # GUI
sivan -g Detector.geo.setup -c Sivan.cfg -f Sim.sim -n # batch
```
Confirm flags in `src/sivan/src/MInterfaceSivan.cxx::Usage()` (`-g -f -c -n -d`).

To read `.sim` programmatically (Python/C++), use `MFileEventsSim` + `MSimEvent` — see
`global-api.md` and `workflows.md` for the event-loop pattern.

---

## RESPONSE — response-matrix generation

Program: `responsecreator` (parallel `mresponsecreator`, distributed `dresponsecreator`).
Library: `libResponseCreator`. Classes prefixed `MResponse`. Source: `src/response/{inc,src}`.
Output: `.rsp`(.gz) N-dimensional matrices (`file-formats.md §7`).

### How it works
`MResponseCreator` loops a `.sim` file, and for each event compares the **simulated truth**
(SIVAN) with the **reconstruction** (REVAN, configured by a `.revan.cfg`) — sometimes also a
`.mimrec.cfg` for imaging responses — and fills multi-dimensional histograms
(`MResponseMatrixON`). `MResponseBuilder` is the base; subclasses define each response type.
Matrices are saved incrementally (every N events) so long runs survive interruption.

### Response types (selected by `-m <mode>`)
Approximate mode codes (verify in `MResponseCreator.cxx` / `--help` — codes have evolved):

| Mode | Builder | Produces |
|------|---------|----------|
| `s` | `MResponseSpectral` | energy redistribution (E_true ↔ E_meas) |
| `il` | `MResponseImaging` | list-mode imaging response (for mimrec) |
| `ib` | `MResponseImagingBinnedMode` | binned imaging response |
| `ic` | `MResponseImagingCodedMask` | coded-mask imaging |
| `a` | `MResponseImagingARM` | ARM resolution matrices |
| `ef` / `en` | `MResponseImagingEfficiency[NearField]` | efficiency (far/near field) |
| `cb` | `MResponseMultipleComptonBayes` | Bayesian Compton-sequence response (for revan CSR) |
| `ct` | `MResponseMultipleComptonTMVA` | TMVA Compton sequencing training data |
| `t` | `MResponseTracking` | electron-tracking response |
| `pb` | `MResponsePolarizationBinnedMode` | polarization response |
| `e` | `MResponseEarthHorizon` | earth-horizon rejection |
| `q`/`qf` | `MResponseEventQuality*` | event-quality PDFs |
| `gt`/`gf` | `MResponseEventClusterizerTMVA*` | event-clustering training |
| `sf` | `MResponseStripPairingTMVAEventFile` | strip pairing |

### Running
```bash
responsecreator -g Detector.geo.setup -f Sim.sim -r MyResponse -m s -c Revan.cfg
responsecreator -g Det.geo.setup -f Sim.sim -r Compton  -m cb -c Revan.cfg -z
responsecreator -g Det.geo.setup -f Sim.sim -r Image    -m ib -c Revan.cfg -b Mimrec.cfg
```
Options (verify in `Usage()`): `-g` geometry, `-f` sim file, `-r` output response name,
`-m` mode, `-c` revan cfg, `-b` mimrec cfg, `-o key=val:key=val` mode options
(e.g. `emin=100:emax=20000:ebins=500`), `-s` save-after-N, `-z` gzip, `-i` max event id.

Parallel/distributed:
```bash
mresponsecreator -g Det.geo.setup -f Sim.p*.inc*.sim -r R -m ib -c Revan.cfg -b Mimrec.cfg
dresponsecreator --datadir=… --targetdir=… -g=… -r=… -m=… -c=… -b=…
```
Merge partial responses and inspect/transform them with **`mresponsemanipulator`**
(`MResponseManipulator`: `statistics`, `append`/join, `collapse`, `divide`, `ratio`,
`probability`).

### Practical chain
1. Simulate a representative set (often a grid of energies/incidence angles, or a broad
   beam) with cosima → `.sim` (use `StoreSimulationInfo all`).
2. Run `responsecreator` with the appropriate `-m` mode + the **same** `.revan.cfg` you'll
   reconstruct real data with.
3. Feed the `.rsp` to mimrec (imaging responses) or revan (Bayesian/TMVA CSR responses).
   The geometry must match between response and data.
