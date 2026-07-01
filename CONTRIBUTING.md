# Contributing

## Before You Start

Read:

- `AGENTS.md`
- `ARCHITECTURE.md`
- `Documentation/DataFoundationRoadmap.md`
- `SECURITY.md`

For modeling or data work, also read `TESTING.md` and `CODE_REVIEW.md`.

## Environment

**Execution context:** ULCO Calculco HPC. Read `Documentation/CalculcoSetup.md`
and export paths via gitignored `.env.calculco` before pipeline work.

Use `uv` from the repository root (`$PROJECT_HOME`):

```bash
uv sync
uv run ruff check .
```

Do not use another dependency manager. Explain any new dependency before adding
it with `uv add` or `uv add --dev`.

## Change Workflow

1. Describe the goal, relevant context, constraints, and definition of done.
2. Write a plan for multi-file, schema, cohort, or modeling changes.
3. Implement one coherent milestone at a time.
4. Add focused tests using synthetic fixtures.
5. Run the smallest relevant checks.
6. Review the diff for data safety, leakage, reproducibility, and stale claims.
7. Update documentation and `CHANGELOG.md`.

Use `WORKFLOWS.md` for task-specific procedures.

## Clinical Data

Never commit or share raw or row-level MIMIC-IV, MIMIC-IV-Note, or eICU data.
Do not include patient records in issues, pull requests, screenshots, test
fixtures, logs, or prompts. Prefer schemas and aggregate statistics.

Generated derived data must stay under ignored directories unless it is a
small, synthetic fixture explicitly created for testing.

## Pull Requests

A pull request should state:

- the research or engineering problem;
- what changed and what remains out of scope;
- data sources and cohort implications;
- leakage and safety considerations;
- verification commands and results;
- generated artifacts, if any, and where they were written; and
- documentation updated.

Do not claim an improvement without a reproducible comparison against an
appropriate baseline and held-out data.

## Commit Guidance

Keep commits focused and reviewable. Do not mix raw data, generated model
artifacts, formatting churn, and behavioral changes in one commit.

## Licensing

No repository-wide open-source license has been selected. Contributions do not
change the licensing status of the project or of third-party datasets.
