# BEvAn: β⁺ Decay Event Analyzer

<p align="center">
  <img src="docs/COSI-SMEX%20Emblem_2E.png" alt="COSI-SMEX mission emblem" width="320">
</p>

<p align="center">
  <a href="https://github.com/cosi-betadecay/BEvAn/actions/workflows/tests.yml"><img src="https://github.com/cosi-betadecay/BEvAn/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/cosi-betadecay/BEvAn/actions/workflows/tests.yml"><img src="https://raw.githubusercontent.com/cosi-betadecay/BEvAn/badges/coverage.svg" alt="Coverage"></a>
  <a href="https://huggingface.co/datasets/aryaraeesi/BEvAn-data"><img src="https://img.shields.io/badge/🤗-Dataset%20used%20during%20R&D-FFD21E" alt="Dataset used during R&D on Hugging Face"></a>
</p>

This project is part of the [Compton Spectrometer and Imager (COSI)](https://cosi.ssl.berkeley.edu/) mission. It flags each simulated event as either β⁺ decay or background, using the event's Compton kinematics and a physics-based likelihood model rather than a black-box classifier.

Positron annihilation following β⁺ decay produces two back-to-back 511 keV photons. Those photons interact mainly through Compton scattering in segmented high-purity germanium (HPGe) detectors, leaving a set of energy deposits and hit positions. By analysing the energy and geometry of those hits, the pipeline reconstructs each event's kinematics and decides whether it is consistent with 511 keV annihilation radiation.

## Requirements

- Python 3.14
- MEGAlib + ROOT, with the `MEGALIB` environment variable set
- A Weights & Biases account and API key (optional — only needed for `--wandb`)

## Setup

MEGAlib and ROOT must be available in your environment before running any of the Python code. MEGAlib's installation guide can be found [here](https://megalibtoolkit.com/setup.html).

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

MEGAlib simulation files live one dataset per folder, `data/{name}/`, holding the train+eval pair `{name}.sim` / `{name}.tra` (for example `data/SPILike/SPILike.sim`) plus the two dedicated prior simulations `{name}_P1.sim` / `{name}_P2.sim`. The detector geometry is passed explicitly via `--geo-file` to the per-dataset entry points (`analysis.py`, `train_model.py`, `inference.py`). The ablation driver takes no geometry flag — it looks each folder's geometry up by name in the static `GEOMETRIES` map in `ablations/harness.py`, so add an entry there for any new dataset.

The dataset used during R&D (6.38 GB — seven geometries, each with its `.sim`/`.tra`
pair, the `mcosima` chunks they point at, and two dedicated prior runs) is hosted on Hugging Face: [**aryaraeesi/BEvAn-data**](https://huggingface.co/datasets/aryaraeesi/BEvAn-data).

```bash
cd data
uvx --from huggingface_hub hf download aryaraeesi/BEvAn-data \
  --repo-type dataset --local-dir .
```

This drops every geometry folder (`SPILike/`, `NCT/`, …) directly into `data/`.
Add `--include "SPILike/*"` to pull a single dataset. Note that `{name}.sim` is an
`mcosima` concatenation file holding **no events** — it points at the
`{name}.inc*.id1.sim.gz` chunks beside it, so a dataset is only usable with its
chunks present.

> **Geometry paths:** use the `$MEGALIB` environment variable form (e.g.
> `$MEGALIB/resource/examples/geomega/special/Max.geo.setup`) — it works
> unquoted, and the tools also expand it internally via `os.path.expandvars`.
> Avoid the `$(MEGALIB)` form on the command line: the shell reads `$(...)` as
> command substitution and blanks it out (it only works inside MEGAlib's own
> config files, where ROOT expands it).

## Running

Analyze a single dataset (train and evaluate in one pass). The class priors are
**never** taken from the training split: `--prior-sim` (required) names one or
more dedicated prior `.sim` files — simulated against the same geometry — whose
events are counted per hit-multiplicity bucket with the ANNI ground truth to
form the per-bucket priors deployed by the classifier:

```bash
python src/BEvAn/analysis.py \
  --geo-file $MEGALIB/resource/examples/geomega/special/SPILike.geo.setup \
  --sim-file data/SPILike/SPILike.sim \
  --tra-file data/SPILike/SPILike.tra \
  --prior-sim data/SPILike/SPILike_P1.sim data/SPILike/SPILike_P2.sim \
  --wandb            # optional
```

### Train once, infer later

To fit a model on a whole simulation and save it to a torch artifact
(`--prior-sim` is required here too; the counted priors and their source files
are recorded in the artifact's provenance):

```bash
python src/BEvAn/train_model.py \
  --geo-file $MEGALIB/resource/examples/geomega/special/SPILike.geo.setup \
  --sim-file data/SPILike/SPILike.sim \
  --tra-file data/SPILike/SPILike.tra \
  --prior-sim data/SPILike/SPILike_P1.sim \
  --out models/SPILike.pt        # default: models/<sim-stem>_<timestamp>.pt
```

Then run real inference with that saved model on any other `.sim`/`.tra` dataset (the model carries the fitted densities and the measured priors that set its operating point; the `.tra` is still back-projected to reconstruct this dataset's source direction for the ARM feature). The result is a torch artifact holding one `uint8` entry per non-empty event of the `.sim`, in file order: `1` = β⁺, `0` = background (events the model cannot score default to `0`):

```bash
python src/BEvAn/inference.py \
  --model models/Max.pt \
  --geo-file $MEGALIB/resource/examples/geomega/special/Max.geo.setup \
  --sim-file data/Max_v2/Max_v2.sim \
  --tra-file data/Max_v2/Max_v2.tra \
  --out predictions/Max_v2.pt    # default: predictions/<sim-stem>_<timestamp>.pt
```

### Ablations

The ablation study isolates one component of the model at a time (CKD hit ordering, learned term weights, per-factor contributions, and the label-window tolerance), each measured against a fair our-model reference:

```bash
python ablations/main.py --wandb   # --wandb optional
```

`scripts/run_ablations.sh` launches the same thing as a detached
background process: nothing prints to the terminal, output goes to
`logs/ablations_<timestamp>.log`, and the process shows up in `top`/`htop` as
`bevan-ablations` instead of `python3`. Extra arguments pass through to the
underlying script.

Results are written to a timestamped folder `ablations/results/<timestamp>/{figures,tables}`;
with `--wandb` the figures are also logged to Weights & Biases.

## Tests

```bash
pytest
```

The suite needs no MEGAlib/ROOT installation: `tests/conftest.py` installs a fake
`ROOT` module (`tests/fake_megalib.py`) before anything imports PyROOT, so it runs
anywhere — including CI, where it runs on every push (see the badge above).

## Documentation

API documentation is built with Sphinx and published to GitHub Pages: [cosi-betadecay.github.io/BEvAn](https://cosi-betadecay.github.io/BEvAn/).
