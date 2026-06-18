# Changelog

All notable repository changes are recorded here. Dates use ISO 8601.

## [Unreleased]

### Added

- Initial `pipeline/` source-inventory skeleton with configuration, safe
  dataset path/header inspection, bounded DuckDB CSV helper, and
  `pipeline.source_inventory` CLI.
- Focused synthetic tests for Milestone 1 configuration and inventory helpers.
- Adult MIMIC-IV/eICU ICU cohort materialization with aggregate manifest,
  source-qualified IDs, first MIMIC ICU stay per admission, eICU top-coded age
  handling, and synthetic cohort tests.
- Aggregate source-table quality profiler with row/key counts, null/cardinality
  metrics, parse/plausibility checks, referential-integrity checks, safe
  scan-failure reporting, and synthetic profiling tests.
- Aggregate EDA dataset-understanding synthesis with stakeholder Markdown
  brief, JSON summary, figure pack, domain-readiness routing, and synthetic
  EDA tests.
- Source-integrity checker for profiling-blocked files with SHA256 manifest
  validation, gzip stream checks, aggregate report output, and synthetic tests.
- Manifest-wide source-integrity mode for checking every file listed in
  configured MIMIC-IV, MIMIC-IV-Note, and eICU `SHA256SUMS.txt` files, with
  source-specific filtering and missing-file reporting.
- `Documentation/CalculcoSetup.md` with verified ULCO Calculco account paths,
  storage layout, completed setup steps, and transfer procedures for username
  `zahmed`.
- Calculco HPC workflow section in `WORKFLOWS.md`.
- Portable agent skills for verification, documentation synchronization, and
  clinical-data safety.
- Conservative Cursor and Codex project configuration.
- A current data-foundation roadmap.
- Citation metadata for the research project.

### Changed

- Rebuilt the README to reflect the actual data-foundation stage.
- Expanded `AGENTS.md` with project-specific data, clinical, verification, and
  review rules.
- Clarified that the Transformer-GNN recommender and grounded explanation
  system are target architecture, not completed clinical software.

### Security

- Explicitly excluded licensed datasets, secrets, model artifacts, and
  generated reports from version control and agent indexing.
