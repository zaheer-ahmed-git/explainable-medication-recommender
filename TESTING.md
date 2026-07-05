# Testing

## Current State

The active working tree contains the source-inventory, adult cohort, aggregate
table-profiling, EDA-summary, source-extraction, and Milestone 5 harmonization
pipeline with focused synthetic tests. Milestone 6 feature and observed-label
builders also have focused synthetic tests. Graph and model modules still need
their own tests as they are added.

Every new active module should arrive with focused tests.

**Execution context:** ULCO Calculco. Export `PROJECT_HOME` and `DATASET_ROOT`
before pipeline CLIs. See `Documentation/Environment.md`.

## Verification Tiers

| Tier | Where | When |
|------|-------|------|
| Lightweight | Calculco login node | Config, unit tests, lint, docs |
| Bounded CLI | Login node (with `DATASET_ROOT`) | Small inventory or manifest checks |
| HPC | OAR compute nodes | Cohort-filtered extraction, large scans |

Do not submit OAR jobs or long-running full-dataset work unless explicitly
requested. See `WORKFLOWS.md` and `scripts/calculco/`.

## Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

During development, prefer a focused path:

```bash
uv run pytest tests/test_config.py tests/test_io_utils.py
uv run pytest tests/test_cohort.py
uv run pytest tests/test_profile_tables.py
uv run pytest tests/test_eda_summary.py
uv run pytest tests/test_source_integrity.py
uv run pytest tests/test_extraction_harmonize.py
uv run pytest tests/test_config.py tests/test_features.py tests/test_build_training_table.py
uv run pytest tests/test_condition_normalization.py
uv run pytest tests/test_condition_mapping_builder.py
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
- Condition normalization preserves source-native tokens, adds shared roll-ups
  only from authoritative fixtures, degrades gracefully when mapping files are
  missing, and drops no diagnosis rows.
- Every artifact records cohort and configuration provenance.

## Full Verification

Before a milestone is considered complete:

1. Run targeted tests.
2. Run the complete test suite.
3. Run Ruff check and format check.
4. Inspect generated manifests and aggregate counts.
5. Review the Git diff.
6. Report checks and any unavailable validation.

Milestone 5 harmonization completion additionally requires the manifest to list
`cohort_stays`, `demographics`, `conditions`, `medications`, `labs`, `vitals`,
`allergies`, `interventions`, and `temporal_events`, with aggregate coverage
and unmapped reports.

Milestone 6 completion additionally requires temporal cutoff tests, censoring
tests, patient split-integrity tests, train-only candidate-catalog tests,
out-of-catalog positive reporting, and aggregate-only manifest checks.
