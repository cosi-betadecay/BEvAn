---
name: code-quality
description: Code-quality workflow for this repo — running ruff lint + format, the single-config setup (ruff.toml is the only ruff config), docstring/import conventions, pytest, and the project discipline of never regressing the champion F1 (0.8935). Use before committing, when cleaning up code, fixing lint errors, adding docstrings, or reviewing a diff for quality.
---

# Code quality for betadecay-analysis

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
  the top-level package set under `src/betadecay-analysis/` —
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
