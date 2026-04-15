# COSI 511 keV Annihilation Analysis

This project is part of the Compton Spectrometer and Imager (COSI) program at the
Space Sciences Laboratory, UC Berkeley. It focuses on detecting and characterizing
511 keV gamma-ray annihilation photons in high-purity germanium (HPGe) detector
data. The goal is to flag each simulated event as either a beta-plus annihilation
signal or background, using Compton kinematics and likelihood-based scoring.

Positron annihilation following beta-plus decay produces two back-to-back 511 keV
photons. In COSI, those photons interact primarily through Compton scattering in
segmented HPGe detectors. Each interaction leaves a set of energy deposits and hit
positions. By analyzing the energy and spatial distribution of hits, the analysis
reconstructs event kinematics and determines whether the event is consistent with
a 511 keV annihilation photon.

This repository uses MEGAlib simulations to generate realistic interaction chains,
labels annihilation truth from the simulated interaction history, and evaluates how
well physics-based likelihoods discriminate beta-plus events from background.
The scientific motivation is twofold: constrain astrophysical sources of 511 keV
emission and validate COSI detector response for annihilation radiation, which
supports both instrument calibration and reconstruction development.

## How events are flagged (beta-plus vs background)

The classifier adapts Zoglauer's Bayesian Compton Sequence Reconstruction (B-CSR,
PhD §4.5.3) to the β⁺-decay vs background problem. For every simulated event the
pipeline produces a ground-truth label and a prediction, then compares the two.

- **Ground truth** (`src/betadecay-analysis/physics/ground_truths.py`): walks the
  MEGAlib interaction chain for each event, locates every `ANNI` process,
  collects the hits descended from its secondaries, and sums their deposited
  energy. The event is labeled β⁺-decay if any annihilation's total deposited
  energy lies within a tolerance of 511 keV. The tolerance is derived from the
  HPGe FWHM (3σ window) in `src/betadecay-analysis/mathematics/calculations.py`.

- **Per-event features**
  (`src/betadecay-analysis/physics/matrix_calculations.py`):
  - `delta_E` — deviation of the total deposited energy from 511 keV.
  - `annihilation_angle` — geometric opening angle derived from hit positions.
  - `arm` — Angular Resolution Measure comparing kinematic and geometric
    scatter angles.

- **Likelihood ratio** (`physics/annihilation_detection_utils.py::R`): following
  Zoglauer eq. 4.26, class-conditional densities are estimated from the training
  events and combined as

  ```
  R = [P_β(ΔE) / P_bg(ΔE)]
    · [P_β(angle | ΔE) / P_bg(angle | ΔE)]
    · [P_β(ARM   | ΔE) / P_bg(ARM   | ΔE)]
  ```

  The 2D joints `P(ΔE, y)` are factored via `conditional_from_joint` so that ΔE
  enters the product exactly once.

- **Posterior and decision**
  (`src/betadecay-analysis/physics/bayesian_annihilation.py`): the empirical
  β vs bg class balance from the training tensors is used as the prior
  `p(β) / p(bg)`, giving `p(β | m) = R · prior / (R · prior + 1)`. Events with
  posterior > 0.5 are predicted β⁺-decay, otherwise background.

- **Metrics**: predictions are compared to ground truth to compute
  TP/FP/FN/TN, precision, recall, false-positive rate, F1, and a confusion
  matrix, all logged to Weights & Biases.

## What this repository does

- Reads MEGAlib `.sim` events through the ROOT-based reader in
  `src/betadecay-analysis/utils/reader_extraction.py`.
- Derives ground-truth β⁺-decay labels by inspecting `ANNI` processes in each
  event's interaction history and matching annihilation hit energies to 511 keV.
- Computes the physics features (ΔE, annihilation angle, ARM) for every event
  that survives `event_data_processing` filtering.
- Builds 1D/2D class-conditional density matrices for β and background using
  shared bin edges, then factors the 2D joints into ΔE marginals and `y | ΔE`
  conditionals.
- Runs per-event Bayesian inference with the empirical class prior and logs
  classification metrics and a confusion-matrix image to W&B.
- Supplies a synthetic-data generator and plotting utilities for sanity-checking
  likelihoods and classifier behavior.
- Documents the MEGAlib `.sim` format (`docs/MEGAlib.md`) and the math behind
  the likelihood-ratio scheme (`docs/math.md`).

## Repository layout

- `src/betadecay-analysis/run.py`: Hydra + W&B entry point; calls
  `annihilation_extractor`.
- `src/betadecay-analysis/physics/annihilation_detection.py`: top-level
  pipeline — feature extraction, tensor assembly, density matrices, prediction.
- `src/betadecay-analysis/physics/annihilation_detection_utils.py`: Zoglauer-style
  likelihood-ratio construction (`R`), per-event lookups, and the `predict`
  routine that logs metrics.
- `src/betadecay-analysis/physics/bayesian_annihilation.py`: posterior
  computation and the β / bg decision rule.
- `src/betadecay-analysis/physics/ground_truths.py`: MEGAlib `ANNI`-based
  truth labeling.
- `src/betadecay-analysis/physics/event_processing.py`: per-event hit
  filtering and preprocessing.
- `src/betadecay-analysis/physics/matrix_calculations.py`: ΔE, annihilation
  angle, ARM, density matrices, and grid lookups.
- `src/betadecay-analysis/mathematics/`: energy-tolerance calculation
  (`calculations.py`) and geometry helpers (`geometry.py`).
- `src/betadecay-analysis/utils/`: ROOT reader setup, plots, and a synthetic
  event generator used by the tests.
- `src/betadecay-analysis/configs/`: Hydra configs (`config.yaml`,
  `wandb.yaml`, `likelihoods.yaml`) controlling setup paths, W&B project, and
  ARM parameters.
- `tests/tests_physics/`: pytest suite for the physics modules.
- `scripts/generate_test_summary.py`: converts pytest JUnit output into a
  Markdown summary for documentation.
- `docs/`: `MEGAlib.md`, `math.md`, and the generated `test-summary.md`.
- `data/`: place MEGAlib simulation files here (e.g., `Activation.sim`).

## Requirements

- Python 3.10
- MEGAlib + ROOT with `MEGALIB` set in your environment
- W&B account and API key (required by the current scripts)

## Setup

```bash
uv sync
source .venv/bin/activate
```

Create a `.env` file with your W&B key so the scripts can log runs:

```text
WANDB_API_KEY=your_key_here
```

## Data

Place MEGAlib `.sim` files in `data/`. The default scripts expect
`data/Activation.sim`. See `docs/MEGAlib.md` for the file format summary.


## Test reporting

All tests are expected to run in a local ROOT-enabled environment. To make the
latest local results easy to reuse in documentation, the recommended workflow is:

```bash
python3 -m pytest --junitxml=artifacts/pytest-junit.xml
python3 scripts/generate_test_summary.py artifacts/pytest-junit.xml --output docs/test-summary.md
```

This keeps the machine-readable source report in `artifacts/pytest-junit.xml`
and writes a README-friendly Markdown summary to
`docs/test-summary.md`. See [docs/test-summary.md](./docs/test-summary.md) for
the latest generated summary.

## References

- MEGAlib documentation and .sim format notes: `docs/MEGAlib.md`
