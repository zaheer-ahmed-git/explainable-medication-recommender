# Testing

## Current State

The active working tree does not yet contain a `tests/` directory or active
pipeline modules. The `pytest` development dependency is available, but a
no-tests result is not evidence that the project is correct.

Every new active module should arrive with focused tests.

## Commands

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

During development, prefer a focused path:

```powershell
uv run pytest tests/test_cohort.py
uv run pytest tests/test_cohort.py -k patient_split
```

## Fixture Policy

- Never use raw MIMIC-IV, MIMIC-IV-Note, or eICU rows in tests.
- Build small synthetic tables with invented identifiers and values.
- Include edge cases deliberately: missing identifiers, repeated stays, null
  units, simultaneous events, and boundary timestamps.
- Keep fixtures readable enough that expected behavior can be reviewed.

## Test Layers

### Unit Tests

Cover normalization, schema validation, deterministic splitting, temporal
windows, candidate generation, metrics, and mapping utilities.

### Contract Tests

Verify required input columns, output schemas, identifier uniqueness, allowed
nulls, units, provenance fields, and manifest structure.

### Integration Tests

Run source-specific extraction and harmonization against tiny synthetic CSV or
Parquet fixtures. Confirm filters are applied before materialization.

### Model Tests

Use small deterministic data to verify feature selection, leakage exclusions,
fit/predict shape, ranking groups, metric calculations, and artifact metadata.
Do not assert that a tiny model reaches clinically meaningful performance.

### Documentation Tests

Check that commands, paths, links, and status statements match the repository.

## Required Invariants

- One patient maps to exactly one split.
- Candidate catalogs are learned from training data only.
- Feature timestamps do not exceed the decision cutoff.
- Future outcomes are excluded from predictors by default.
- Source-qualified identifiers remain unique.
- Harmonization reports unmapped concepts instead of silently dropping them.
- Every artifact records cohort and configuration provenance.

## Full Verification

Before a milestone is considered complete:

1. Run targeted tests.
2. Run the complete test suite.
3. Run Ruff check and format check.
4. Inspect generated manifests and aggregate counts.
5. Review the Git diff.
6. Report checks and any unavailable validation.
