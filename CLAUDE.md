# CLAUDE.md

Project-specific guidance for Claude Code when working in this repository.

## Project

COSI 511 keV annihilation analysis. Classifies MEGAlib-simulated events as
β⁺-decay annihilation signal vs background using Compton kinematics and a
Zoglauer-style Bayesian likelihood-ratio scheme (B-CSR, PhD §4.5.3). See
`README.md` for the science overview and `docs/math.md` for the math.

## Environment

- Python 3.10 (pinned in `pyproject.toml`).
- Requires MEGAlib + ROOT with `MEGALIB` set in the shell environment.
- Dependencies managed with `uv`. Setup: `uv sync && source .venv/bin/activate`.
- W&B logging is required by the entry-point scripts. `WANDB_API_KEY` must be
  in `.env`.
- Default input data: `data/Activation.sim` (MEGAlib `.sim` format,
  documented in `docs/MEGAlib.md`).

## Repository layout

- `src/betadecay-analysis/run.py` — Hydra + W&B entry point.
- `src/betadecay-analysis/configs/` — Hydra configs (`config.yaml`,
  `wandb.yaml`, `likelihoods.yaml`).
- `src/betadecay-analysis/physics/` — event processing, ground truths,
  Compton-cone reconstruction, physics factors.
- `src/betadecay-analysis/modeling/` — likelihood-ratio construction,
  Bayesian posterior, density-matrix building.
- `src/betadecay-analysis/pipeline/` — `train.py` and `eval.py` orchestration.
- `src/betadecay-analysis/mathematics/` — energy-tolerance and geometry helpers.
- `src/betadecay-analysis/dataset/` — dataset assembly.
- `src/betadecay-analysis/utils/` — ROOT reader, plots, calculations.
- `tests/tests_general/`, `tests/tests_specialized/` — pytest suites.
- `scripts/generate_test_summary.py` — JUnit → Markdown summary.
- `docs/` — `MEGAlib.md`, `math.md`, `event_reconstruction.md`,
  `test-summary.md`.

Note: README still references the older flat physics-only layout; the active
code has been split across `physics/`, `modeling/`, and `pipeline/`. Prefer
reading the directory tree over the README's layout section when navigating.

## Style and tooling

- Ruff is the formatter/linter. Line length 100. Select rules `E,F,W,D,I`
  with Google-style pydocstyle. Tests are exempt from `D` rules.
- isort first-party packages: `physics`, `mathematics`, `utils`, `experiments`.
- Keep Google-style docstrings on public functions in `src/`.

## Tests

Tests need a ROOT-enabled environment. Standard workflow:

```bash
python3 -m pytest --junitxml=artifacts/pytest-junit.xml
python3 scripts/generate_test_summary.py artifacts/pytest-junit.xml \
  --output docs/test-summary.md
```

## Working notes

- The branch `60-use-native-megalib-code` is the current working branch;
  `main` is the PR base.
- Don't introduce mocks for ROOT/MEGAlib in tests — the suite is intended to
  run against a real ROOT install.
- The synthetic event generator in `utils/` is the right tool for unit-level
  sanity checks of likelihoods and classifier behavior.