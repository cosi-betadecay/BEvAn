# β⁺ Decay Signatures in COSI: 511 keV Annihilation Detection

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
