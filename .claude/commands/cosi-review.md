---
description: Run the multi-agent code review (orchestrator → code-quality + physics + megalib + performance reviewers) over a diff, PR, branch, or files. Add `full` to review the whole repo instead of just the diff.
argument-hint: "[optional: files, a PR number, a git ref, or `full` — defaults to the current branch vs main]"
---

Use the **review-orchestrator** agent (`.claude/agents/review-orchestrator.md`) to run
a full multi-agent code review of this repo.

Scope: $ARGUMENTS

**If the scope above contains `full`** (e.g. `/cosi-review full`), do **not** scope to a
diff — review the **entire repository's source** instead: every `.py` file under
`src/betadecay-analysis/` and `ablations/`. Each specialist audits the whole codebase in
its lane, not just changed lines. Any words besides `full` still apply as an extra scope
hint (e.g. `/cosi-review full physics` → whole-repo review weighted toward the physics
lane). This is the deep periodic audit, not the per-change gate.

Otherwise, if the scope is empty, review the current branch's changes —
`git diff main...HEAD` plus any uncommitted work in the working tree.

The orchestrator will:
1. Determine the exact set of changed `.py` files for the scope.
2. Dispatch the four specialist reviewers **in parallel** — `code-quality-reviewer`,
   `physics-reviewer`, `megalib-reviewer`, `performance-reviewer` — each using its own skill(s).
3. Merge, deduplicate, and prioritize their findings (Blocker → Nit), surfacing any
   change that could regress the champion F1 or silently alter pipeline behavior.
4. Return one report ending in a **SHIP / SHIP WITH FIXES / DO NOT SHIP** verdict.

This is a **read-only** review — report findings only; do not edit code unless I
explicitly ask afterward. If the environment forbids the orchestrator from spawning
subagents, it will fall back to running the four review passes itself and say so.
