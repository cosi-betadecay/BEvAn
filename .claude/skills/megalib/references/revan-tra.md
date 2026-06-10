# revan & the .tra format

revan = MEGAlib's event reconstruction tool: raw hits (.sim/.evta) →
reconstructed physical events (.tra). This project invokes it as
`revan -g <geometry> -f <simfile> -n -a` (utils/reader_extraction.py).

## CLI (src/revan/src/MInterfaceRevan.cxx, ParseCommandLine)

| Flag | Meaning |
|---|---|
| `-n` / `--no-gui` | batch mode (no GUI, ROOT batch) |
| `-a` / `--analyze` | analyze the input file immediately, then exit |
| `-f <file>` | input (.sim, .sim.gz, .evta) |
| `-g <file>` | geometry (.geo.setup) |
| `-o <file>` | output .tra name (default: input with .tra suffix) |
| `-c <file.cfg>` | revan configuration (default ~/.revan.cfg) |
| `-C Key=Value` | override single config value (repeatable) |
| `--oi` | write Original Input (truth start) info into the .tra |
| `-s` | generate spectra; `-d` debug; `-t` test run |

NOTE: with no `-c`, revan uses `~/.revan.cfg` if present — reconstruction
parameters are therefore machine-state, not repo-state. Pin a config file
for reproducibility.

## Reconstruction pipeline (src/revan/src/MRawEventAnalyzer.cxx)

1. Coincidence search (time window)
2. Event clustering, then hit clustering (per-detector min-distance)
3. Electron tracking
4. **Compton Sequence Reconstruction (CSR)**: Classic (χ²), Bayesian, or
   TMVA neural net — `CSRAlgorithm` in the cfg (1 = Classic). Default
   `CSRMaxNHits` = 7 (cf. this repo's 7-hit subset cap!)
5. Decay detection

## .tra format (serialization in src/global/misc/src/M*Event.cxx)

Header (`Type tra`, `Version`, `Geometry`, ...) then per-event blocks:

Common keywords: `SE` (start), `ET <type>` (event type), `ID <n>`
(**= originating sim event ID**), `TI <time>`, `BD <reason>` (bad flag),
`DC` (decay flag), `OI x y z dx dy dz px py pz E` (truth start info, only
with `--oi`), `CC` comments, `EN` (end of file).

Event types (`ET`): `CO` Compton, `PH` photo-absorption, `PA` pair,
`MU` muon, `DY` decay, `PT` PET, `MT` multi, `UN` unidentifiable.

### ET PH (photo) — MPhotoEvent.cxx
- `PE <energy>` total absorbed energy [keV]
- `PP <x> <y> <z>` interaction position
- `PW <weight>`

### ET CO (Compton) — MComptonEvent.cxx
- `CE Eg dEg Ee dEe` — scattered-γ and recoil-e⁻ energies ± errors
- `CD C1x C1y C1z dC1... C2x C2y C2z dC2... Dex Dey Dez dDe...` — first
  and second interaction positions and electron direction, with errors
- `SQ <n>` sequence length; `CT q1 q2` Compton quality factors;
  `PQ` clustering quality; `TL`/`TE`/`TQ` track length/initial
  deposit/quality; `TF tof dtof`; `LA <lever arm>`; `CW <window>`
- `CH h x y z E t ...` per-hit cluster lines

### ET PA (pair) — MPairEvent.cxx
- `PC x y z` creation position; `PE`/`PP` electron/positron energy ±
  error + direction; `PI` initial deposit; `TQ` track quality

## sim ↔ tra ID mapping

`MRERawEvent::ConvertToPhysicalEvent()` copies the raw event ID, so a .tra
event's `ID` is the sim event's first `ID` field. revan silently DROPS
events it cannot reconstruct or that fail selections — .tra IDs are a
strict subset of sim IDs with gaps (e.g. in this repo's data, sim event 1
is absent from Activation.tra; the first tra event is ID 2). Never assume
positional alignment between the two files; join on ID.
