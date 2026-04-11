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

The core pipeline in `physics/annihilation_detection.py` takes each simulated
event and produces two labels: beta-plus annihilation (beta-decay signal, bdecay)
or background (bg).

- Ground truth: uses the MEGAlib interaction chain to find ANNI processes,
  collects their secondary hits, sums the deposited energies, and checks whether
  the total falls within the 511 keV window defined by
  `mathematics/calculations.py` (3-sigma tolerance derived from HPGe FWHM).
- Prediction: scores all hit combinations using Compton-kinematic likelihoods
  and flags the event as beta-plus if the best score exceeds a threshold; otherwise it is labeled background.

Scoring uses the following components:

- Energy consistency: a Gaussian-like weight on the sum of hit energies relative
  to 511 keV (`physics/likelihoods/energy.py`).
- ARM (angular resolution measure): compares geometric and kinematic scatter
  angles when at least three hits are available
  (`physics/likelihoods/arm.py`).
- Klein-Nishina weight: optional Compton cross-section weighting for two or more
  hits (`physics/likelihoods/kn.py`).

The final score takes the maximum over all hit combinations to allow for
uncertainty in interaction ordering. The predicted label is then compared to the
ground-truth label to compute precision, recall, false-positive rate, and F1.

## What this repository does

- Reads MEGAlib simulation events and marks ground truth annihilation signals
  by tracking ANNI interactions and summing associated hit energies.
- Scores hit combinations using energy consistency and angular resolution measure
  (ARM), with optional Klein-Nishina (KN) weighting to detect candidate 511 keV
  events.
- Flags events as beta-plus or background and logs performance metrics
  (precision, recall, false-positive rate, F1) and confusion matrices to
  Weights & Biases (W&B).
- Provides tuning scripts to explore likelihood distributions and classifier
  behavior for ARM, KN, and posterior models.
- Documents the MEGAlib .sim format and expected data layout.

## Repository layout

- `main.py`: run the end-to-end annihilation detection pipeline with W&B logging.
- `physics/annihilation_detection.py`: ground-truth labeling, detection scoring,
  and metric computation.
- `physics/likelihoods/`: ARM, KN, Compton-kinematic, and energy likelihoods.
- `physics/posterior.py`: posterior scoring utilities.
- `mathematics/calculations.py`: energy-tolerance calculation (based on HPGe FWHM).
- `utils/reader_extraction.py`: MEGAlib/ROOT event reader setup.
- `utils/plots.py`: plotting and summary statistics for likelihoods.
- `experiments/`: tuning scripts for ARM, KN, and posterior behavior.
- `docs/MEGAlib.md`: overview of MEGAlib .sim files.
- `data/`: place simulation files (e.g., `Activation.sim`) here.

## Requirements

- Python 3.10
- MEGAlib + ROOT with `MEGALIB` set in your environment
- W&B account and API key (required by the current scripts)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file with your W&B key so the scripts can log runs:

```text
WANDB_API_KEY=your_key_here
```

## Data

Place MEGAlib `.sim` files in `data/`. The default scripts expect
`data/Activation.sim`. See `docs/MEGAlib.md` for the file format summary.

## Running the analysis

The main entry point runs the detector simulation analysis and logs metrics:

```bash
python main.py
```

By default, `main.py` uses the MEGAlib example geometry file:
`$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup` and
`data/Activation.sim`. Adjust these paths or call
`physics.annihilation_detection.annihilation_extractor` directly to analyze
other datasets or tune the scoring weights.

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

## Experiments and tuning

- `experiments/arm_tuning.py`: explores ARM, energy, and KN likelihood behavior.
- `experiments/kn_tuning.py`: evaluates KN likelihood distributions.
- `experiments/posterior_tuning.py`: compares posterior scores for signal vs. background.

These scripts log distribution diagnostics (histograms, CDFs, violin plots) to W&B.

## Notes on methodology and future work

- Energy tolerance is derived from HPGe resolution (FWHM -> 3-sigma window) in
  `mathematics/calculations.py`. Sweeping tolerance windows can be done by
  adjusting that function or exposing it as a parameter in analysis scripts.
- The current pipeline is physics-driven (Compton kinematics + likelihoods).
  Future work includes machine learning models (e.g., CNNs or graph networks)
  that can learn nonlinear correlations among hit energies and positions and
  complement physics-constrained reconstruction.

## References

- MEGAlib documentation and .sim format notes: `docs/MEGAlib.md`
