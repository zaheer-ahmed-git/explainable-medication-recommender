# Workflows

## Significant Change

1. Read the closest instructions and canonical architecture documents.
2. State the goal, constraints, assumptions, and definition of done.
3. Create a milestone plan using `PLANS.md`.
4. Implement one milestone.
5. Run focused verification.
6. Review the diff using `CODE_REVIEW.md`.
7. Update affected docs and `CHANGELOG.md`.

## Build or Change a Cohort

1. Define source population, inclusion criteria, exclusion criteria, unit of
   analysis, index time, and deduplication rule.
2. Implement source-specific logic before unified logic.
3. Materialize only identifiers and necessary cohort fields first.
4. Produce a manifest with source counts and filter attrition.
5. Test key uniqueness, referential integrity, patient overlap, and boundary
   conditions.
6. Compare counts with documented dataset scales without treating approximate
   public counts as exact acceptance criteria.
7. Update `Documentation/DataFoundationRoadmap.md`.

Current broad adult cohort command:

```powershell
uv run python -m pipeline.cohort
```

The generated cohort Parquet files are local ignored artifacts under
`Dataset/processed/cohorts/`; only aggregate manifest counts should be reported
outside the local environment.

## Profile a Large Table

1. Inspect the header and metadata without reading rows into logs.
2. Run a bounded schema profile.
3. Select required columns.
4. Use DuckDB or chunked streaming for complete scans.
5. Save aggregate results under `reports/`.
6. Review results for identifiers, units, nulls, duplicates, impossible values,
   and source-specific conventions.

Current aggregate profile command:

```powershell
uv run python -m pipeline.profile_tables
```

The generated `reports/quality_profile.json` is an ignored local artifact. It
must contain aggregate counts and column metrics only, never row samples or note
text.

## Build EDA Brief

1. Confirm `reports/source_inventory.json`, `reports/cohort_manifest.json`, and
   `reports/quality_profile.json` exist.
2. Run `uv run python -m pipeline.eda_summary`.
3. Review `reports/eda_dataset_understanding.md` for stakeholder-facing
   messages, quality blockers, and next actions.
4. Review `reports/figures/` for aggregate charts.
5. Do not add row examples, note text, identifiers, or clinical
   recommendations to EDA outputs.

## Check Source Integrity

1. Run `uv run python -m pipeline.source_integrity` for profiling-blocked files.
2. Review `reports/source_integrity_failed_tables.json`.
3. Treat checksum mismatches or gzip failures as source-integrity blockers.
4. Re-transfer or re-download affected files before extraction or feature
   engineering.
5. Consider CSV parser fallbacks only after checksum and gzip validation pass.

## Build Source Inventory

1. Inspect only file metadata and CSV headers.
2. Run `uv run python -m pipeline.source_inventory`.
3. Confirm `reports/source_inventory.json` remains ignored.
4. Check missing expected files and checksum-file presence.
5. Do not print or paste clinical rows, note text, identifiers, or free-text
   values from the source files.

## Add an Extraction Module

1. Write the output schema and provenance fields first.
2. Validate required source columns.
3. Filter to cohort identifiers at the source query.
4. Normalize names only after preserving original values.
5. Add synthetic contract tests.
6. Record row counts and mapping coverage.

## Build a Training Table

1. Freeze cohort, index time, feature window, label window, and patient split.
2. Build candidates from the training partition only.
3. Create observed-positive and sampled/implicit-negative labels with explicit
   caveats.
4. Exclude future and leakage-prone features by default.
5. Validate one patient belongs to one split.
6. Write a schema and manifest with class balance and candidate coverage.

## Run an Experiment

1. Name the hypothesis and baseline.
2. Freeze data and configuration identifiers.
3. Run on validation data while developing.
4. Use test or external-validation data only for the final locked evaluation.
5. Save configuration, metrics, seed, feature list, and model version.
6. Report failures, uncertainty, subgroup limitations, and negative results.
7. Do not promote a poster illustration or exploratory result to a validated
   clinical claim.

## Fix a Bug

1. Reproduce the behavior on synthetic or minimized data.
2. Add a regression test that fails.
3. Apply the smallest correction.
4. Rerun the reproduction and targeted tests.
5. Check whether the bug affected prior artifacts or reported metrics.
6. Update documentation or changelog when behavior changed.

## Documentation-Only Change

1. Verify claims against the current tree.
2. Keep one canonical source for each fact.
3. Fix links and commands.
4. Distinguish current state from roadmap.
5. Run Markdown/link checks available in the repository and inspect the diff.

## Dependency Change

1. Explain the concrete need.
2. Prefer the standard library or an existing dependency when reasonable.
3. Use `uv add`, `uv add --dev`, or `uv remove`.
4. Inspect `pyproject.toml` and `uv.lock`.
5. Run focused tests and lint.
6. Note security, licensing, and reproducibility implications.

## Calculco HPC (ULCO)

Use the cluster for heavy extraction, modeling, and long-running jobs. Canonical
server reference: `Documentation/CalculcoSetup.md`.

1. SSH as `zahmed@calculco.univ-littoral.fr`.
2. Keep code in `/nfs/home/lisic/zahmed/ResearchModule`.
3. Keep licensed clinical data only under
   `/nfs/data/protected/lisic/zahmed/ResearchModule/Dataset`.
4. Use `/workdir/lisic/zahmed/runs` for temporary job I/O; copy results back to protected storage after jobs complete.
5. Transfer large files with `rsync` via `pcsdata.univ-littoral.fr`.
6. Submit compute with OAR (`oarsub -S script.sh`), not on the login node.
7. Load software with `module load` or a conda/uv environment as documented in `Documentation/CalculcoSetup.md`.
