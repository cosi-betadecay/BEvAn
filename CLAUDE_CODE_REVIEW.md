# Claude Code Review Protocol

This document defines how code reviews are performed in this repository using
Claude. Reviews are **agentic**: Claude dispatches a set of specialized agents,
each with a narrow focus, and aggregates their findings into a single report.

## Review rule (read first)

**When this file is used to drive a code review, Claude reviews only. Claude
does NOT modify code, configs, tests, or documentation unless the user
explicitly asks for changes.** The output of a review is a report, not a
diff. If an agent identifies an issue, it describes the issue and suggests a
fix in prose — it does not apply the fix. The user decides which findings, if
any, become change requests in a follow-up turn.

This applies even when the fix is trivial (typo, obvious bug, one-line
correction). The reviewer and the implementer are intentionally separated so
the user retains full control over which changes land.

## Review scope

By default, a review covers the code paths relevant to the user's request
(e.g. "review the classifier pipeline", "review `annihilation_detection.py`").
If no scope is given, Claude asks before starting rather than reviewing the
entire repository.

## The three review agents

Each review dispatches the following agents in parallel. Each agent returns a
written report of its findings, which Claude aggregates into a single
consolidated review.

### 1. Nuclear astrophysics agent

**Focus:** physical correctness.

Checks that the physics used in the code matches what is claimed in
docstrings, comments, and commit messages, and is consistent with the
underlying detector (COSI / MEGA-style HPGe Compton telescope) and the
science goal (β⁺-decay vs background classification).

Typical questions this agent answers:
- Are energy windows, tolerances, and reference values (e.g. 511 keV,
  1022 keV, HPGe FWHM-derived 3σ tolerances) physically justified?
- Do the derived quantities (ΔE, annihilation angle, ARM, Compton scatter
  angles) match their standard definitions in the Compton-imaging
  literature?
- Are conservation laws respected (energy, momentum, photon kinematics)?
- Does the ground-truth label (`ground_truth_bdecay`) correctly identify
  the physical signature it claims to identify, including corner cases
  (e.g. fully-contained 1022 keV events, partially-escaped photons)?
- Are Klein-Nishina, absorption probabilities, and angular-resolution
  quantities used in the regimes where they apply?
- Are units consistent across the pipeline (keV vs MeV, radians vs
  degrees, cm vs mm)?

This agent does not review code style or Python correctness — only
physics.

### 2. Bug agent

**Focus:** correctness of the code itself.

Checks for logic errors, off-by-one mistakes, incorrect tensor shapes,
broken edge cases, silently swallowed exceptions, misused APIs (ROOT,
PyTorch, NumPy, Hydra, W&B), mismatches between a function's
docstring/signature and its behavior, and any place where the code
computes something other than what the caller would reasonably expect.

Typical questions this agent answers:
- Are tensor operations doing what they claim (broadcasting, indexing,
  `bucketize` direction, `index_put_` accumulation)?
- Do functions handle NaN, empty tensors, zero-length events, and
  out-of-support lookups in a way that's consistent with their
  docstrings?
- Are the Bayesian posterior calculations in
  `bayesian_annihilation.py` numerically and algebraically correct
  (e.g. variable names matching the quantities they hold)?
- Are Hydra configs actually wired to the code paths they claim to
  control?
- Are tests measuring what they claim to measure, or is any assertion
  trivially satisfied?
- Are there dead code paths, unreachable branches, or TODO comments
  referencing resolved issues?

This agent does not review physics correctness or performance — only
whether the code does what it says.

### 3. Memory agent

**Focus:** memory footprint and efficiency.

Checks how much memory the pipeline consumes at peak, where the
hotspots are, whether large intermediate tensors can be avoided, and
whether the memory profile scales acceptably with event count and
bin count.

Typical questions this agent answers:
- What is the peak memory cost of density matrices at the current
  default bin count (2000×2000)? Does it scale linearly, quadratically,
  or worse with `n_bins`?
- Are there places where tensors are copied unnecessarily (e.g.
  `.to(...)` without a device change, repeated `.flatten()` on
  already-flat inputs)?
- Could any per-event list be replaced with a pre-sized tensor filled
  in place?
- Are there opportunities to stream events through the pipeline instead
  of materializing the full lists before building tensors?
- Does the ROOT reader release event objects correctly
  (`SetOwnership`), and are there any leaks across the event loop?
- Do the test fixtures (particularly session-scoped ones in
  `conftest.py`) duplicate large tensors that could be shared?

This agent also suggests concrete ways to reduce memory where it's
excessive — as prose recommendations, not edits.

## Review output format

Each review produces a single aggregated report with four sections:

1. **Scope** — what was reviewed (files, functions, time window).
2. **Nuclear astrophysics findings** — from agent 1.
3. **Bug findings** — from agent 2.
4. **Memory findings** — from agent 3.

Each finding includes:
- A short title.
- The file and line(s) it refers to.
- Severity (high / medium / low).
- A description of the issue.
- A suggested direction for fixing it (prose, not code).

Findings are ordered by severity within each section.

## How to invoke a review

The user requests a review by referencing this file, e.g.:

> "Run the code review from CLAUDE_CODE_REVIEW.md on
> `physics/annihilation_detection_utils.py`."

Claude then:
1. Confirms the scope.
2. Dispatches the three agents in parallel against that scope.
3. Aggregates their reports into the format above.
4. **Stops.** No edits, no commits, no config changes.

If the user wants any of the findings implemented, they make that request
in a separate turn.
