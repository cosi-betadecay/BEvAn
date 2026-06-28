---
name: code-conventions
description: How a well-formed function, class, and module looks in betadecay-analysis — the static authoring conventions (signatures, type hints, Google docstrings, the NaN-return idiom, naming, tensor shape comments, dataclasses, module constants) plus an adapted "Power of 10" discipline (single-responsibility functions, bounded size/complexity, assert at seams), anchored to the Google Python Style Guide + PEP 8. ADVISORY ONLY: this skill makes recommendations for a human to apply — the assistant must NOT edit code under it. Use when writing or reviewing a function/class, or when asked how code here should be structured. For the *workflow* (running ruff, the cascade, the F1 discipline) see the code-quality skill.
---

# Code conventions for betadecay-analysis

This is the **reference** for what good code *looks* like here — the static
authoring conventions, distilled from the existing code, not invented. For the
*workflow* (how to run ruff/pytest, tracing the cascade, the champion-F1
discipline) see the **code-quality** skill; the two are complementary.

## Rule for the assistant: recommend, never edit

**This skill is advisory.** Under it the assistant **produces recommendations
only — it does not modify source files.** When asked to "apply conventions",
"clean up", or "fix" code while operating under this skill:

- Report what deviates and *propose* the change (file:line · what · why · the
  suggested diff in a fenced block) — but **leave the edit to the human.**
- Do **not** run `Edit`/`Write` on `.py` files, do **not** add `# noqa`, and do
  **not** change `ruff.toml` to enforce these rules. Enabling enforcement or
  editing code is a human decision.
- If the human explicitly says "go ahead and make the change," that is a separate
  instruction outside this skill — follow the code-quality skill's discipline
  then (measure F1 for anything touching the cascade).

The conventions below are the standard to *recommend against*, not a license to
rewrite.

## Rule 0 — repo invariants and F1 beat generic style (read first)

Every rule below is a *default*, not a law. This is a research classifier with a
characterized ~0.89 ceiling and a **champion eval F1 of 0.8935**. Where a generic
convention ("a function does one thing", "no duplication", "shorter is better")
collides with a documented pipeline invariant, **the invariant wins** and gets a
comment saying why. Load-bearing examples that *look* like things you'd "clean up"
but must not — flag them as intentional, never recommend removing them:

- **`calculate_tolerance` is deliberately shared** by the 511 keV label *and* the
  511-aware feature push. That is single-source-of-truth, not accidental coupling.
- **The ΔE double-count in the n3 bucket** looks redundant; de-duping it cost F1
  0.8935 → 0.8735. It stays.
- **NaN returns are load-bearing** (see below) — never recommend "cleaning them
  up" into raises or zero-fills.

So: the conventions apply freely to *new* and *pure-style* code; for anything that
touches feature values, bucket routing, the label, or the prior counts, any
recommendation must come with the code-quality skill's "trace the cascade, measure
F1" caveat.

## Authoring conventions (what the code already does)

### Functions

- **Full type hints, always** — every parameter and the return. Physics knobs are
  keyword args with defaults (`n_std: int = 3`, `ordering: str = "ckd"`). Tensors
  are `torch.Tensor`; plain records are typed (`-> list[float]`, `-> dict[...]`).
- **Google-style docstrings, enforced** (ruff `D`, `convention = "google"`): a
  one-line summary, a blank line, then prose and `Args:` / `Returns:`. `__init__`
  needs its own docstring (D107); multi-line docstrings need the summary-then-blank
  shape (D205). Module/package docstrings are *not* required (`D100`/`D104` off),
  but the readers carry one when it documents a format/unit contract (see
  `sim_text_reader.py`).
- **Math notation in docstrings is intentional**, not a typo — `σ`, `φ`, `×`, `−`,
  `α` are whitelisted (`allowed-confusables`). Keep `m_e c²`, `FWHM = 2.355·σ`,
  `511 keV` spelled the physics way.
- **Naming**: `snake_case` functions; physics-notation locals kept verbatim
  (`v_in`, `v_out`, `L_in`, `L_out`, `x0`, `x1`, `cos_theta_geo`) — readability for
  a physicist beats a generic rename. Leading `_` for module-private helpers
  (`_draw_density_panel`).
- **Shape comments on tensors**: annotate non-obvious tensor shapes inline
  (`x0 = pos[:, 0, :]  # (B, 3)`) and state shapes in the docstring
  (`positions: (B, n, 3) hit positions`). Clamp floors are explicit and commented
  (`torch.clamp(..., min=1e-3)`).
- **Single source of truth**: compute a shared quantity once and import it; don't
  re-derive a constant or threshold in a second place.

### The NaN-return idiom (load-bearing — never recommend "fixing")

A feature function returns **NaN** to mean *"this event has no usable subset for
this feature"* — a routing signal, not an error. `feature_bucket` routes each
event by *which* of `delta_E` / `arm` / `annihilation_angle` are finite. So:

- Returning `float("nan")` / a NaN tensor for "not available" is correct; raising,
  imputing, or zero-filling would silently misroute the event into a different
  bucket model and shift that bucket's prior.
- A guard clause that flips a NaN to finite (or vice versa) is a **behavioral**
  change, not style — flag it for F1 re-measurement, don't wave it through.

### Classes

- **Docstring + typed `__init__` with its own docstring.** Bind plain attributes in
  `__init__`; document load-bearing constructor knobs with trailing inline comments
  (`self.joint_smoothing = joint_smoothing  # Laplace pseudo-counts for the 2D
  joints`).
- **Module-level constants in CAPS**, defined once near the top with a comment
  explaining the contract (`BUCKETS = (1, 2, 3)`, `FEATURES = (...)`,
  `SAVE_FORMAT_VERSION = 1`). Buckets/feature names live in `datasets.py` — import
  them, don't re-list.
- **A `config()`-style reproducibility method** where a class has tunable state
  (`Trainer.config()` returns the exact kwargs to rebuild it; that dict is what
  `save`/`load` records). A new tunable field should be threaded through `config()`.
- **`@dataclass` for plain records** (`SimIA`, `SimHit`, `SimTextEvent`): one field
  per line, each with a unit/semantics comment (`energy: float  # keV`), mutable
  defaults via `field(default_factory=list)`.
- **Docstring-only stub classes** for the PyROOT API (`utils/megalib_types.py`):
  thin typed interfaces listing the `M.*` accessors actually used.

### Modules

- One concern per module, matching the `dataset / physics / modeling / pipeline /
  utils` split (CLAUDE.md "Layout"). Imports isort-sorted (ruff `I`,
  `combine-as-imports`); first-party is the top-level package set
  (`dataset`, `physics`, `modeling`, `mathematics`, `pipeline`, `utils`) — not
  `src`. Section-banner comments (`####…`) group related functions within a module.

## The "Power of 10", adapted for Python research code

NASA/JPL's *Power of 10* was written for **safety-critical embedded C**. Several
rules are about C's footguns and don't exist in Python — transcribing them
literally would be cargo-culting. The honest adaptation, anchored to the **Google
Python Style Guide** and **PEP 8** (both already in effect via ruff) rather than
the C originals:

| JPL rule (C) | Here |
|---|---|
| 1. Simple control flow, no recursion | **Keep.** Flat control flow; recursion is absent and should stay rare. |
| 2. Loops have fixed bounds | **Keep (spirit).** No unbounded `while True`; event loops are data-bounded. |
| 3. No heap allocation after init | **Drop** (no manual memory). Weak echo: preallocate tensors, don't grow them in hot loops. |
| 4. Functions ≤ ~one page (~60 lines) | **Keep — the atomicity rule.** One function, one job; if you can't summarize it in one docstring line, it's a split candidate. |
| 5. ≥2 runtime assertions per function | **Adapt.** Drop the rigid count; keep CLAUDE.md's rule: *prefer a cheap `assert` on dtype / shape / finiteness at a seam over a comment.* |
| 6. Smallest data scope | **Keep.** Narrow locals; no needless module globals; private helpers get a leading `_`. |
| 7. Check every return value / validate params | **Keep.** Type hints + guard clauses; don't silently ignore a returned value. |
| 8. Limit the preprocessor | **Drop** (none). Echo: limit metaprogramming / monkeypatching / dynamic attribute magic. |
| 9. Limit pointer use | **Drop** (none). Echo: avoid surprising mutable aliasing and in-place tensor mutation. |
| 10. Zero warnings, all checks on | **Keep — direct.** `ruff check` stays clean; no blanket `# noqa`. A suppression needs a specific code *and* a reason. |

**Single responsibility / atomicity** is the headline: a function does one thing,
at one level of abstraction, nameable in one verb phrase.

## Size & complexity — recommendation thresholds (not enforced)

These are **review heuristics for a human to apply**, not active lint rules. They
correspond to the ruff rules a maintainer *could* enable (`C901`, `PLR0913`,
`PLR0915`) — but this skill does **not** turn them on or add `# noqa`; that's a
deliberate human choice, separate from authoring guidance.

- **Complexity** ≤ ~10 branches (`C901`/mccabe).
- **Arguments** ≤ ~5 (`PLR0913`) — past that, the usual fix is a small `@dataclass`
  config object (safe for code off the cascade, e.g. plotting/CLI).
- **Statements** ≤ ~50 per function (`PLR0915`).

### Current functions over these thresholds (recommendation targets)

Use this list when *recommending* cleanups — split into "leave alone" vs. "worth
proposing":

**Load-bearing — recommend leaving as-is (don't propose a refactor without F1):**

- `physics/event_processing.py` `event_data_processing` — complexity 24,
  70 statements. Core cascade (hit-subset construction + CKD/energy ordering);
  complex because the physics is. Splitting risks every feature in every bucket.
- `modeling/matrix_calculations.py` `build_density_matrix` (11 args) and
  `pipeline/model_selection.py` `search_hyperparams` (12 args) — density
  estimation + the bin-size/threshold search; args feed cascade seams.

**Off the cascade — reasonable to *propose* a cleanup (no F1 impact):**

- `utils/plots.py` `_draw_density_panel` (9 args), `_draw_density_1d_panel` (7) —
  pure presentation; a `@dataclass` panel-config is the textbook fix.
- `analysis.py` `main` — complexity 11, 53 statements, 6 args; barely over.
  Extracting arg-parse/setup is safe.
- `train_model.py` `train_and_save` (7 args) — CLI wrapper.
- `modeling/matrix_calculations.py` `build_density_matrix_1d` (6),
  `physics/compton_cone_reconstruction.py` `FarFieldImager.__init__` (6),
  `pipeline/model_selection.py` `calibrate_global_offset` (6),
  `pipeline/train.py` `add_2d` (8) — mild; case-by-case.

(Counts measured against ruff defaults 10 / 5 / 50; refresh with
`ruff check src --select C901,PLR0913,PLR0915` if the code changes.)

## Quick review checklist (recommend against this; don't auto-fix)

1. Typed signature + Google docstring (summary line, blank, `Args`/`Returns`)?
2. Does it do **one** thing nameable in one phrase? (else a split candidate, unless
   it's a documented cascade exception)
3. NaN means "not available" and is preserved — no impute/raise/zero-fill?
4. Assert dtype/shape/finiteness at the seam instead of a comment?
5. Tensor shapes annotated; constants in CAPS / single-sourced?
6. Within the size heuristics, or a justified exception?
7. If it touches features/buckets/label/prior — flag for F1 re-measurement
   (code-quality skill).
