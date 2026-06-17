---
name: code-change-verification
description: Verify a non-trivial repository change with focused tests, Ruff, configuration parsing, diff review, and a concise result report.
---

# Code Change Verification

## Inputs

- The intended behavior.
- The changed files.
- Any user-specified acceptance checks.

## Steps

1. Read `AGENTS.md` and `TESTING.md`.
2. Inspect the changed files and identify the smallest meaningful test surface.
3. Check whether the change affects data safety, cohort logic, temporal logic,
   labels, splits, schemas, metrics, or research claims.
4. Run focused tests first.
5. Run `uv run ruff check .` for Python or configuration changes.
6. Run `uv run ruff format --check .` when Python formatting is relevant.
7. Parse TOML or JSON configuration files with an appropriate standard parser.
8. Run `git diff --check` and inspect the final diff.
9. Distinguish regressions from pre-existing failures or missing test coverage.

## Success Criteria

- Relevant checks pass.
- Raw clinical data and generated artifacts remain untracked.
- Data-leakage and reproducibility risks were reviewed.
- The final report lists commands, results, and remaining gaps.

## Failure Handling

Do not hide failures. Report the failing command, the important error, whether
it appears related to the change, and the smallest next correction.
