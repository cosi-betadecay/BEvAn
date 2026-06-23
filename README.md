# COSI 511 keV Annihilation Analysis

<p align="center">
  <img src="docs/COSI-SMEX%20Emblem_2E.png" alt="COSI-SMEX mission emblem" width="320">
</p>

This project is part of the Compton Spectrometer and Imager (COSI) program at the
Space Sciences Laboratory, UC Berkeley. It flags each simulated COSI event as
either a β⁺ (positron) annihilation signal or background, using the event's
Compton kinematics and a physics-based likelihood model rather than a black-box
classifier.

Positron annihilation following β⁺ decay produces two back-to-back 511 keV
photons. In COSI those photons interact mainly through Compton scattering in
segmented high-purity germanium (HPGe) detectors, leaving a set of energy
deposits and hit positions. By analysing the energy and geometry of those hits,
the pipeline reconstructs each event's kinematics and decides whether it is
consistent with 511 keV annihilation radiation.

## What is novel here

- **Physics-based, not learned.** Discrimination comes from a Bayesian
  likelihood ratio built directly out of detector physics — deviation from
  511 keV, the Angular Resolution Measure (ARM), and a back-to-back
  annihilation score — so every decision is interpretable in physical terms.
  The scheme adapts Zoglauer's Bayesian Compton Sequence Reconstruction
  (B-CSR) to the β⁺-vs-background problem.
- **It reuses COSI's own reconstruction.** The source direction that the ARM
  feature is measured against comes from MEGAlib's native far-field Compton
  backprojection (`MBackprojectionFarField`), not a hand-rolled imager — so the
  analysis stays consistent with the instrument's standard event reconstruction.
- **Multiplicity-aware modelling.** Events are routed by how many usable hits
  they produce into separate buckets, each with its own generative model and
  class prior, so an event is only ever scored on features it actually has.
- **Detector-agnostic by construction.** The same pipeline runs over any
  MEGAlib `.sim`/`.tra` pair and reads each detector's geometry straight from
  the file header, so the method can be exercised across COSI-family
  geometries without code changes.

## How it works

For every dataset the pipeline runs four stages:

1. **Compton reconstruction** — backproject the Compton events in the `.tra`
   file into an accumulated sky image and take the peak as the source direction.
2. **Feature extraction** — stream the `.sim` events, label each one β⁺ /
   background from its `ANNI` interaction history (a 511 keV photopeak criterion
   set by the HPGe energy resolution), and compute its physics features.
3. **Training** — estimate per-bucket class-conditional densities and class
   priors on a reproducible training split.
4. **Evaluation** — combine the per-event likelihood ratio with the prior into a
   Bayesian posterior, decide signal vs background, and report the metrics.

## Requirements

- Python 3.10
- MEGAlib + ROOT, with the `MEGALIB` environment variable set
- A Weights & Biases account and API key (optional — only needed for `--wandb`)

## Setup

MEGAlib and ROOT must be available in your environment before running anything
Python — source MEGAlib's setup script so its libraries are on the path:

```bash
source "$MEGALIB/bin/source-megalib.sh"
```

Then create the Python environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

To enable Weights & Biases logging (the optional `--wandb` flag), create a
`.env` file with your key:

```text
WANDB_API_KEY=your_key_here
```

## Data

Place MEGAlib simulation files in `data/` as matching `{name}.sim` / `{name}.tra`
pairs (for example `data/SPILike.sim` and `data/SPILike.tra`). The detector
geometry is read automatically from each file's header, so no separate geometry
path is required for the batch runner.

## Running

Run a single dataset:

```bash
python src/betadecay-analysis/run.py \
  --sim-file data/SPILike.sim \
  --tra-file data/SPILike.tra \
  --wandb            # optional
```

Or run every `.sim`/`.tra` pair found in `data/` and write a per-dataset summary
to `results/<timestamp>.csv`:

```bash
python src/betadecay-analysis/total_run.py --wandb   # --wandb optional
```
