---
name: code-quality
description: Code-quality workflow for this repo — running ruff lint + format, the single-config setup (ruff.toml is the only ruff config), docstring/import conventions, pytest, tracing how a change in one function cascades downstream through the pipeline (feature → bucket routing → density → likelihood ratio → F1), and the project discipline of never regressing the champion F1 (0.8935). Use before committing, when cleaning up code, fixing lint errors, adding docstrings, changing a function the pipeline depends on, or reviewing a diff for quality.
---

# Code quality for BEvAn

Python 3.10 project (`requires-python == 3.10.*`), linted and formatted with
**ruff**, tested with **pytest**. Tooling lives in `.venv/`; always invoke via
`.venv/bin/<tool>` (or `uv run <tool>`), never a global install.

## Config: single source of truth

Ruff is configured by **`ruff.toml` (repo root) only**. The `[tool.ruff]` block
was removed from `pyproject.toml` — a standalone `ruff.toml` takes precedence
and would silently override it, so all ruff config lives in `ruff.toml`. Don't
re-add a `[tool.ruff]` block to `pyproject.toml`; it would be a silent trap.

Effective settings (verify with `ruff check --show-settings`):

- line-length **110**, target **py310** (matches `requires-python`/`.python-version`).
- Rule set `E W F I UP B A C4 TID SIM Q RUF D`.
- Docstrings (`D`) **enforced**, **google** convention; `D100`/`D104`
  (module/package docstrings) ignored. `E501` deferred to the formatter.

## The commands

```bash
.venv/bin/ruff check src                # lint
.venv/bin/ruff check --fix src          # auto-fix safe violations
.venv/bin/ruff format src               # format (E501 is deferred to formatter)
.venv/bin/ruff check --diff src         # preview fixes without writing
.venv/bin/pytest                        # tests
.venv/bin/pytest --cov                  # tests + coverage (pytest-cov installed)
```

Before any commit touching `.py`: `ruff check --fix` then `ruff format`, then
re-run `ruff check` to confirm clean. The tree currently lints **clean** — keep
it that way; don't let a diff *add* violations. (There is no `tests/` tree at
the moment; the test suite was removed in an earlier cleanup.)

## Conventions

- **Imports**: isort via ruff `I`; `combine-as-imports = true`. First-party is
  the top-level package set under `src/BEvAn/` —
  `known-first-party = ["dataset", "physics", "modeling", "mathematics", "pipeline", "utils"]`
  (NOT `"src"`; that dir is what lands on `sys.path`, not `src`). Keep this list
  in sync if a top-level package is added or renamed.
- **Line length**: 110. E501 is ignored at lint time and handled by
  `ruff format` — let the formatter wrap; don't hand-wrap.
- **Quotes / comprehensions / simplify**: `Q`, `C4`, `SIM`, `B`, `UP` are on —
  prefer the idioms ruff suggests (f-strings, dict/set comprehensions,
  `contextlib.suppress`, modern type syntax) rather than fighting the linter.
- **Docstrings**: enforced, google-style. Multi-line docstrings need a one-line
  summary followed by a blank line (D205); `__init__` needs its own docstring
  (D107). Module/package docstrings are not required (`D100`/`D104` ignored).

## Trace the cascade: one change ripples downstream

The pipeline is a **cascade**, and the dangerous part is that most of the seams
fail *silently* — a change produces plausible-but-different numbers, not an error.
Before editing any function the pipeline consumes, **follow the data all the way to
the F1 number**, not just the immediate call site. The data flow:

```
read .sim/.tra (reader_extraction, sim_text_reader)
  → event_processing  (hit subsets, ordering = "ckd" | "energy")
  → physics_factors   (delta_E, arm, annihilation_angle)  +  ground_truths (label)
  → Datasets.feature_bucket  (routes each event by WHICH features are finite)
  → split_dataset     (train / eval)
  → model_selection.search_hyperparams (bin sizes) + Trainer.fit (densities)  [TRAIN only]
  → calibrate_global_offset (decision threshold)                              [TRAIN only]
  → Evaluator / calculate_probabilities.R (likelihood ratio) + class prior (counts)
  → confusion_counts → metrics (F1 / AUC)
```

### Known cascade points (where a "local" edit silently moves F1)

1. **Feature finiteness → bucket routing.** `feature_bucket` routes events by which
   of `delta_E`/`arm`/`annihilation_angle` are *finite* (NaN = "not available").
   A guard clause in a feature function that returns finite where it used to return
   NaN (or vice versa) **re-partitions events across buckets** — they get scored by
   a *different* bucket model *and* shift each bucket's prior. NaN is load-bearing;
   never impute or zero-fill it to "clean up".
2. **`calculate_tolerance` is shared by the label AND the features.** It defines the
   511 keV label in `ground_truths`, *and* the 511-aware energy push inside `arm` /
   `annihilation_angle`. Change it once and **labels and features move together** —
   which is exactly why the `gt_tolerance` ablation re-extracts the pool per `n_std`.
3. **`ordering` ("ckd" vs "energy").** Feeds subset construction → which subset
   minimizes `|feature|` → the feature *values* → the densities. Changing the
   default ordering changes every feature in every bucket.
4. **NaN-parked events still count toward the class prior.** Events with no usable
   processing sit in bucket 1 with NaN: excluded from the confusion matrix but still
   counted in `n_beta_decay`/`n_bg`. So changing what gets excluded shifts the prior,
   which moves the decision boundary for **all** events.
5. **Binning / `rho_floor` / smoothing** (`matrix_calculations`, `model_selection`)
   → density shape → LLR → threshold calibration → operating point. The bin search
   overfits at high `n_iter` (documented — guard floors exist for a reason).
6. **`EPS` / log floors are duplicated** in `calculate_probabilities.R` and
   `ablations/harness`. Change one without the other and the ablation silently
   diverges from production.
7. **Term order within a bucket model** feeds `R()` positionally (and the ablation's
   `term_llr_columns`). Reordering terms silently misassigns densities to factors.

### How to trace before you edit

- **Find the consumers, not just the callers.** `rg "calculate_tolerance" src`,
  `rg "feature_bucket|isfinite" src`, `rg 'ordering=' src` — see everywhere the
  symbol's *output* flows, including `ablations/`.
- Ask "does this change which events are finite / which bucket they land in / what
  the label is / what the prior counts?" If yes, it is **behavioral**, not style.
- End the trace at the metric. A change is only "safe" once the eval F1/AUC is
  unchanged — reasoning that it "should be" equivalent is not enough (see below).

## Project discipline (matters more than lint)

This is a research classifier. The **champion eval F1 is 0.8935** (commit
86bb140) and sits at a characterized ~0.89 ceiling. The hard rule from the
user's working style: **a cleanup or refactor must not regress F1.** Several
"obvious" simplifications here are load-bearing (e.g. the ΔE double-count in
the n3 bucket — de-duping it cost F1 0.8935→0.8735).

So "code quality" here means:

1. Style/lint cleanups (ruff) — always safe, do freely.
2. Behavioral refactors — **measure F1 before and after** via the eval
   pipeline; revert if it moves the number. Never assume a "harmless" change is
   harmless. Parser-vs-production noising differences also matter — a result
   on the text parser is not confirmed until checked on the production reader.

When in doubt, separate pure-style commits (no behavior change) from
behavioral ones so a regression is bisectable.
