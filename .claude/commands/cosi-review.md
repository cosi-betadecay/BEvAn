---
description: Run the multi-agent code review (orchestrator → code-quality + physics + megalib reviewers) over a diff, PR, branch, or files.
argument-hint: "[optional: files, a PR number, or a git ref — defaults to the current branch vs main]"
---

Use the **review-orchestrator** agent (`.claude/agents/review-orchestrator.md`) to run
a full multi-agent code review of this repo.

Scope: $ARGUMENTS

If the scope above is empty, review the current branch's changes — `git diff main...HEAD`
plus any uncommitted work in the working tree.

The orchestrator will:
1. Determine the exact set of changed `.py` files for the scope.
2. Dispatch the three specialist reviewers **in parallel** — `code-quality-reviewer`,
   `physics-reviewer`, `megalib-reviewer` — each using its own skill(s).
3. Merge, deduplicate, and prioritize their findings (Blocker → Nit), surfacing any
   change that could regress the champion F1 or silently alter pipeline behavior.
4. Return one report ending in a **SHIP / SHIP WITH FIXES / DO NOT SHIP** verdict.

This is a **read-only** review — report findings only; do not edit code unless I
explicitly ask afterward. If the environment forbids the orchestrator from spawning
subagents, it will fall back to running the three review passes itself and say so.
