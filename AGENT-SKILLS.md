# Agent Skills

Repository skills live under `.agents/skills/` and are available to compatible
coding agents through progressive disclosure.

## code-change-verification

Use after a non-trivial code or configuration change.

Inputs:

- changed files;
- intended behavior; and
- relevant test surface.

Outputs:

- checks selected and run;
- concise results;
- distinction between regressions and pre-existing failures; and
- remaining verification gaps.

## docs-sync

Use when code, commands, architecture, project status, or research claims
change.

Outputs:

- the smallest required documentation updates;
- corrected links and commands; and
- explicit separation of planned, implemented, evaluated, and deprecated work.

## research-data-safety

Use for tasks that inspect, transform, summarize, or model clinical datasets.

Outputs:

- a bounded data-access approach;
- confirmation that raw rows are not exposed;
- provenance and leakage checks; and
- safe locations for generated artifacts.

## Adding a Skill

Create a skill only after a workflow is repeated and stable. Each skill should:

- have one clear job;
- include `name` and `description` frontmatter;
- use imperative steps;
- declare inputs, outputs, and success criteria; and
- avoid scripts unless deterministic automation is genuinely needed.
