---
name: code-quality
description: Code-quality workflow for this repo — running ruff lint + format, the dual-config gotcha (root ruff.toml silently overrides pyproject [tool.ruff]), docstring/import conventions, pytest, and the project discipline of never regressing the champion F1 (0.8935). Use before committing, when cleaning up code, fixing lint errors, adding docstrings, or reviewing a diff for quality.
---

# Code quality for betadecay-analysis

Python 3.10 project (`requires-python == 3.10.*`), linted and formatted with
**ruff**, tested with **pytest**. Tooling lives in `.venv/`; always invoke via
`.venv/bin/<tool>` (or `uv run <tool>`), never a global install.

## ⚠ Dual-config gotcha (read first)

There are **two** ruff configs and they conflict:

- `ruff.toml` (repo root) — line-length **110**, target **py312**, rule set
  `E F I UP B A C4 TID SIM Q RUF`, **no docstring (D) rules**.
- `pyproject.toml [tool.ruff]` — line-length 100, target py311, rule set
  `E F W D I` with google docstring convention.

Ruff uses **only one**: a standalone `ruff.toml` takes precedence and the
`[tool.ruff]` block in `pyproject.toml` is **silently ignored**. Verified:
`ruff check --show-settings` resolves to line-length 110 / py312 / preview off.

**So the effective rules are `ruff.toml`** — docstring enforcement (`D`) and
the google convention in pyproject are NOT active. Don't trust the pyproject
block when reasoning about what will pass CI. If you intend docstrings to be
enforced, that's a config fix (consolidate to one file), not a per-file effort.

## The commands

```bash
.venv/bin/ruff check src tests          # lint
.venv/bin/ruff check --fix src tests    # auto-fix safe violations
.venv/bin/ruff format src tests         # format (E501 is deferred to formatter)
.venv/bin/ruff check --diff src tests   # preview fixes without writing
.venv/bin/pytest                        # tests
.venv/bin/pytest --cov                  # tests + coverage (pytest-cov installed)
```

Before any commit touching `.py`: `ruff check --fix` then `ruff format`, then
re-run `ruff check` to confirm clean. There are baseline lint errors in the
tree (~11 at last count) — when cleaning, fix what you touch; don't let a diff
*add* new violations.

## Conventions

- **Imports**: isort via ruff `I`; `known-first-party = ["src"]`,
  `combine-as-imports = true`. First-party = the `src/` package tree
  (`pipeline`, `dataset`, `physics`, `modeling`, `mathematics`, `utils`).
- **Line length**: 110 (effective). E501 is ignored at lint time and handled
  by `ruff format` — let the formatter wrap; don't hand-wrap to 100.
- **Quotes / comprehensions / simplify**: `Q`, `C4`, `SIM`, `B`, `UP` are on —
  prefer the idioms ruff suggests (f-strings, dict/set comprehensions,
  `contextlib.suppress`, modern type syntax) rather than fighting the linter.
- **Docstrings**: not enforced by the active config, but match surrounding
  style (google-style is the project's intent per `pyproject.toml` and
  `pydocstyle.convention`).

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
