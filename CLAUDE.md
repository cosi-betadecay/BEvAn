# gstack
Use the /browse skill from gstack for all web browsing. Never use mcp__claude-in-chrome__* tools.

Available gstack skills (invoke with /skill-name):
- /browse — browser automation and web browsing
- /design — HTML/visual design tools
- /design-html — HTML design variant
- /design-shotgun — multi-variant design generation
- /review — code review
- /plan-design-review — plan + design review
- /autoplan — automatic planning
- /pair-agent — pair programming agent
- /benchmark — benchmarking
- /investigate — investigation/research
- /learn — learning assistant
- /freeze / /unfreeze — freeze/unfreeze codebase changes
- /checkpoint — save a checkpoint
- /careful — cautious mode
- /canary — canary deployment checks
- /qa / /qa-only — QA testing
- /health — project health check
- /retro — retrospective
- /codex — code documentation
- /gstack-upgrade — upgrade gstack itself

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
