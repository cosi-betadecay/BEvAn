# SKILLS.md

Skills available to agents working in this repo: project-local skills live in
`.claude/skills/`, pinned third-party skills in `.agents/skills/` as recorded
in `skills-lock.json` (all from the `wentorai/research-plugins` GitHub source).

## Project-local skills (not in the lockfile)

- **megalib** (`.claude/skills/megalib/SKILL.md`) — reference for MEGAlib,
  extracted from the local source tree (`~/Documents/COSI/megalib`) and
  verified against it: .sim/.tra formats, IA/ANNI/HTsim field maps,
  particle IDs, the PyROOT bindings, cosima/revan/geomega, and how this
  repo's pipeline consumes them. Deep material lives in
  `.claude/skills/megalib/references/`. Maintained by hand; update when
  the data format or pipeline changes.
- **code-quality** (`.claude/skills/code-quality/SKILL.md`) — code-quality
  workflow: ruff lint/format commands, the dual-config gotcha (root
  `ruff.toml` silently overrides `pyproject.toml [tool.ruff]`), import/style
  conventions, pytest, and the discipline of never regressing the champion
  F1 (0.8935) during cleanups. Use before committing or when reviewing a diff.
- **compton-physics** (`.claude/skills/compton-physics/SKILL.md`) — the physics
  reference behind the classifier, sourced entirely from Kierans, Takahashi &
  Kanbach 2022 ("Compton Telescopes for Gamma-ray Astrophysics",
  arXiv:2208.07819): Compton kinematics, Klein-Nishina, CKD/Compton sequencing,
  the Compton Data Space + ARM, SPD/electron tracking, Doppler broadening,
  sensitivity, polarimetry, MeV backgrounds, and the COSI instrument. Maps each
  equation to the repo's `delta_E` / `arm` / `annihilation_angle` features and
  `theta_geo`/`theta_kin`/CKD ordering. Deep material in
  `.claude/skills/compton-physics/references/`. Use when reasoning about why a
  feature is physically motivated or what the 511 keV / 1022 keV lines mean.
- **positron-annihilation-511** (`.claude/skills/positron-annihilation-511/SKILL.md`)
  — the astrophysics of the 511 keV line (the signal itself), sourced entirely
  from Prantzos et al. 2011 (Rev. Mod. Phys. 83, 1001; arXiv:1009.4620):
  positronium (para/ortho, the f_Ps fraction, why most annihilations are *not* a
  clean back-to-back 2γ), the annihilation channels and spectrum, the β⁺-decay
  source isotopes (²⁶Al, ⁴⁴Ti, ⁵⁶Co, ²²Na — the same activation radioactivities
  behind COSI's background), positron energies, line widths, energy loss /
  thermalization / annihilation, and Galactic morphology. Deep material in
  `.claude/skills/positron-annihilation-511/references/`. Use when reasoning
  about the 511 keV / 1022 keV signal, the annihilation-angle feature, the energy
  tolerance, positronium, or what physically produces signal vs background.
- **gpu-performance** (`.claude/skills/gpu-performance/SKILL.md`) — GPU-friendliness,
  vectorization, and time/space-complexity workflow for the torch pipeline: profile
  first, vectorize the per-event scalar loops in `pipeline/datasets.py`, device/dtype/sync
  hygiene (incl. the per-event `empty_cache` and scalar `float()` anti-patterns already in
  the tree), Big-O/space reasoning per stage, and the F1-equivalence test every speedup
  must pass. The engineering counterpart to `code-quality`; defers torch mechanics to the
  `pytorch` skill. Use on the GPU-friendliness branch or any perf/vectorization diff.
- **pytorch** (`.claude/skills/pytorch/SKILL.md`) — third-party reference
  (`itsmostafa/llm-engineering-skills`, added via skillfish) for torch mechanics:
  `torch.profiler`, `torch.compile` (inductor vs cudagraphs), gradient checkpointing,
  data-loader bottlenecks, device patterns. Used for the generic torch *how*; see
  `gpu-performance` for what's load-bearing in *this* pipeline.

## Code-review agents (`.claude/agents/`)

A multi-agent code-review workflow. Invoke with the **`/cosi-review`** slash command
(`.claude/commands/cosi-review.md`) or by asking to "use the review-orchestrator".
All reviewers are read-only and share one finding contract
(Severity · file:line · What · Why · Fix).

- **review-orchestrator** (`.claude/agents/review-orchestrator.md`) — scopes the
  change (diff/PR/branch/files), dispatches the four specialists in parallel,
  merges/dedupes/prioritizes, and gives a SHIP / SHIP WITH FIXES / DO NOT SHIP
  verdict. Falls back to running the passes itself if nested subagents are disallowed.
- **code-quality-reviewer** (`.claude/agents/code-quality-reviewer.md`) — uses the
  **code-quality** + **code-conventions** skills: lint/conventions, the pipeline
  cascade, F1-regression risk.
- **physics-reviewer** (`.claude/agents/physics-reviewer.md`) — uses
  **compton-physics** + **positron-annihilation-511**: feature/kinematics correctness
  against the physics.
- **megalib-reviewer** (`.claude/agents/megalib-reviewer.md`) — uses **megalib**:
  `.sim`/`.tra` parsing, ANNI ground truth, PyROOT API, sim↔tra ID joins.
- **performance-reviewer** (`.claude/agents/performance-reviewer.md`) — uses
  **gpu-performance** + **pytorch**: GPU-friendliness, vectorization, time/space
  complexity, and the proof that a speedup left the champion F1 unchanged.

## Domain skills

- **astrophysics-data-guide** (`skills/domains/physics/astrophysics-data-guide`)
  — astrophysics dataset handling. Relevant for COSI simulation and analysis
  workflows.
- **nasa-ads-api** (`skills/domains/physics/nasa-ads-api`) — querying NASA
  ADS for literature (e.g., 511 keV emission, Compton reconstruction refs).
- **math-skills** (`skills/domains/math`) — general mathematics support,
  useful alongside `docs/math.md` for the likelihood-ratio derivations.
- **algorithms-complexity-guide** (`skills/domains/cs/algorithms-complexity-guide`)
  — algorithm design and complexity analysis.

## AI / ML skills

- **ai-ml-skills** (`skills/domains/ai-ml`) — umbrella ML methodology.
- **pytorch-guide** (`skills/domains/ai-ml/pytorch-guide`) — PyTorch
  (project pins `torch==2.1.2`).
- **pytorch-lightning-guide** (`skills/domains/ai-ml/pytorch-lightning-guide`)
  — Lightning patterns for training/eval loops.

## Research & analysis skills

- **claude-scientific-guide** (`skills/research/methodology/claude-scientific-guide`)
  — scientific methodology for Claude-driven research.
- **deep-research-skills** (`skills/research/deep-research`) — multi-source
  research workflows.
- **ai-scientist-v2-guide** (`skills/research/automation/ai-scientist-v2-guide`)
  — automated scientific workflow (AI Scientist v2).
- **mle-agent-guide** (`skills/research/automation/mle-agent-guide`) — MLE
  agent automation.
- **statistics-skills** (`skills/analysis/statistics`) — statistical analysis,
  directly relevant to the Bayesian classifier and density estimation here.

## Updating

Pinned skills are version-locked by `computedHash` in `skills-lock.json`. To
change one, update that file (typically via the tooling that produced it)
rather than editing this document by hand. Keep this file in sync with the
lockfile when entries are added or removed. Project-local skills are edited
directly in `.claude/skills/`.
