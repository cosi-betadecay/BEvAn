# COSI β⁺ Classifier — "Tagging the Positron Sky"

Physics-based likelihood classifier that tags each simulated COSI event as a β⁺ /
positron-annihilation signal (back-to-back 511 keV photons) or background, using
the event's Compton kinematics rather than a black-box model.

> **Scope guardrail:** this is an **interpretable, physics-based naive-Bayes
> likelihood-ratio model** — per-bucket histogram densities, not a neural net.
> No deep learning, no gradient-trained classifier, no learned feature embeddings.
> The model already does the only tuning it should: a small, *guarded* per-bucket
> bin-size search (`pipeline/model_selection.py`) and a decision-threshold
> calibration. Keep both conservative — both have hard floors precisely so noise
> can't move the operating point. If a change implies deep learning, a black-box
> classifier, or replacing the interpretable physics features, stop and flag it.

## Environment

- Python **3.10** exactly (`==3.10.*`), managed with **uv** (`pyproject.toml` +
  `uv.lock`) — not requirements.txt / conda.
- Requires **MEGAlib + ROOT** with the `MEGALIB` environment variable set; the
  pipeline reads `.sim`/`.tra` event files through PyROOT (`import ROOT as M`).
- Activate:
  ```bash
  source "$MEGALIB/bin/source-megalib.sh"   # ROOT/MEGAlib on the path first
  uv sync && source .venv/bin/activate
  ```
- Weights & Biases logging is **optional** (`--wandb`, needs `WANDB_API_KEY` in `.env`).

## Commands

Entry scripts live in `src/betadecay-analysis/` and take geometry/data explicitly:

- Analyze one dataset (train + eval in one pass):
  `python src/betadecay-analysis/analysis.py --geo-file $MEGALIB/.../SPILike.geo.setup --sim-file data/SPILike.sim --tra-file data/SPILike.tra`
- Batch every `.sim`/`.tra` pair in `data/` → `results/<timestamp>.csv`:
  `python src/betadecay-analysis/batch_analysis.py` (geometry looked up by name in its `GEOMETRIES` map)
- Train once, save a torch artifact: `python src/betadecay-analysis/train_model.py ... --out models/<name>.pt`
- Infer with a saved model: `python src/betadecay-analysis/inference.py --model models/<name>.pt ...`
- Ablation study: `python ablations/main.py` (writes `ablations/results/<timestamp>/{figures,tables}`)
- Lint/format: `ruff check` / `ruff format` (config in `ruff.toml`, see below)
- Tests: `pytest` (pythonpath is wired to `src/betadecay-analysis/`; no suite exists yet)

## Layout

- `src/betadecay-analysis/dataset/`    — streams the MEGAlib reader into per-event,
  per-class, per-bucket feature tensors; train/eval split
- `src/betadecay-analysis/physics/`    — features (`physics_factors.py`: delta_E,
  511-aware ARM, annihilation angle), `event_processing`, Compton-cone
  reconstruction, and `ground_truths` (the 511 keV label)
- `src/betadecay-analysis/modeling/`   — histogram density estimation
  (`matrix_calculations.py`), the likelihood ratio (`calculate_probablities.py`),
  Bayesian decision (`bayesian_annihilation.py`)
- `src/betadecay-analysis/pipeline/`   — `train`, `eval` (metrics, thresholds),
  `model_selection` (bin-size search + threshold calibration)
- `src/betadecay-analysis/mathematics/`— Compton geometry (`theta_geo`, `theta_kin`)
- `src/betadecay-analysis/utils/`      — readers, plots, W&B, local CSV results
- `ablations/`                         — the ablation study (its own driver + harness)
- `data/`                              — `{name}.sim` / `{name}.tra` pairs + cosima sources
- `results/`, `models/`                — per-run CSV outputs; saved model artifacts

## Pipeline contract — read before touching feature or model code

The pipeline is a cascade: **read `.sim`/`.tra` → features → fit per-bucket
densities → likelihood ratio → decision → eval**. Failures here rarely raise; they
produce plausible-but-wrong numbers that propagate silently. Guard the seams:

- **NaN is load-bearing, not an error.** A NaN feature means *"this event has no
  usable subset for this feature."* Events are routed to a hit-multiplicity bucket
  by *which features are finite* (`Datasets.feature_bucket`): bucket 1 = delta_E
  only, 2 = +ARM, 3 = +annihilation. NaN-ratio events are counted toward the class
  prior but `excluded` from the confusion matrix. Do **not** impute or zero-fill
  NaN — that would silently misroute events and poison a bucket's density.
- **Label and bucket are co-located per event.** They are decided together in the
  same loop (`compute_event_features`), so a row can never be mislabeled relative to
  its bucket. Preserve that invariant — don't relabel or re-bucket in a separate pass.
- **Units are fixed: energy in keV, angles in radians.** 511 / 1022 keV and the
  resolution tolerance are all keV; ARM normalizes the angular residual by π. A
  unit/convention mismatch between stages is the dominant silent failure mode.
- **Densities are fit on the train split only.** Bin edges, density matrices, and
  threshold calibration must come from train (or train-internal CV folds), never the
  full set — otherwise eval leaks into the model.
- **Term order is fixed between fit and predict.** The likelihood ratio multiplies
  term densities positionally; keep each bucket's term list in the same order.

Prefer a cheap `assert` on dtype / shape / finiteness at a seam over a comment.

## Where domain knowledge lives (skills)

Project-local skills (`.claude/skills/`, see `SKILLS.md` for the full set incl.
pinned domain skills):

- **megalib** — `.sim`/`.tra` formats, IA/ANNI/HTsim fields, PyROOT bindings,
  cosima/revan/geomega, and how this repo consumes them.
- **code-quality** — ruff lint/format workflow, the dual-config gotcha (`ruff.toml`
  silently overrides any `[tool.ruff]` in `pyproject.toml`), import/style/docstring
  conventions, pytest, and **tracing the cascade** (how a change in one function
  ripples downstream to the F1 number).
- **compton-physics** — how COSI *detects* the photons: Compton kinematics,
  Klein-Nishina, `theta_geo`/`theta_kin` (= φ^geo/φ^kin), ARM, CKD ordering, the
  511/1022 keV constants. Source: Kierans, Takahashi & Kanbach 2022.
- **positron-annihilation-511** — what the photons *are*: the 511 keV line,
  positronium (why a clean back-to-back 2γ is the minority, f_Ps ≈ 0.97), the β⁺
  source isotopes, positron energies, line widths. Source: Prantzos et al. 2011.

Relevant pinned skills include `statistics-skills` (density estimation / the
Bayesian model), `astrophysics-data-guide`, and `nasa-ads-api` (511 keV / Compton
literature).

## Code-review agents (`.claude/agents/`)

A multi-agent code-review workflow, invoked with **`/cosi-review`** (or by asking
to "use the review-orchestrator"):

- **review-orchestrator** — scopes the change, dispatches the three specialists in
  parallel, merges/dedupes/prioritizes their findings, and gives a
  SHIP / SHIP WITH FIXES / DO NOT SHIP verdict. Read-only.
- **code-quality-reviewer** — lint/conventions + the cascade analysis + F1-regression
  risk (uses the `code-quality` skill).
- **physics-reviewer** — physics correctness of the features/kinematics (uses
  `compton-physics` + `positron-annihilation-511`).
- **megalib-reviewer** — `.sim`/`.tra` parsing, ANNI ground truth, PyROOT API (uses
  `megalib`).

All reviewers are read-only (they report findings, they don't edit). They share one
finding contract (Severity · file:line · What · Why · Fix) so the orchestrator can
merge them. The orchestrator falls back to running the three passes itself if its
environment forbids nested subagents.

## Project memory (`.claude/memory/`)

Auto-loaded assistant memory, version-controlled in this repo so it syncs across
machines (this Mac ↔ the Linux MEGAlib VM). `MEMORY.md` is the index — one line per
entry; each entry is its own `<slug>.md` file (anchor state, champion, how-to-work
feedback, don't-retry guardrails). Curated deliberately: keep only load-bearing
facts, fold rather than fragment, prune stale entries.

The harness reads this from `~/.claude/projects/<slug>/memory/`, which is a
**symlink** to `.claude/memory/` here — that's what makes it auto-load each session.
One-time setup per machine (the slug differs because it's derived from the repo's
absolute path):

```bash
# find <slug> with: ls ~/.claude/projects/
ln -s "$(pwd)/.claude/memory" ~/.claude/projects/<slug>/memory
```

After that it travels with normal `git pull` / `push`; new memories appear as
untracked changes under `.claude/memory/`.

## Conventions

- Results are written to timestamped folders: `results/<timestamp>.csv` (batch) and
  `ablations/results/<timestamp>/{figures,tables}`. Random splits/folds use seeded
  `torch.Generator`s for reproducibility.
- Don't change results while "just cleaning up" — check the new run's CSV against the
  previous run before accepting a refactor. Feature/model edits must be re-measured,
  not assumed neutral.
- Ablations each isolate one component (calibration, CKD ordering, learned weights,
  factor contributions, label-window tolerance) and are matched to a fair our-model
  reference. Keep them self-contained in `ablations/`.
