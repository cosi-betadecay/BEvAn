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
