# Changelog

All notable repository changes are recorded here. Dates use ISO 8601.

## [Unreleased]

### Added

- Semantic condition normalization layer in `pipeline/harmonize.py`: optional,
  gracefully degrading shared roll-up tokens (CCSR, CCS, ICD-9→ICD-10 GEM
  crosswalk, ICD chapter, structural ICD category), a conservative eICU
  diagnosis-text dictionary, and a sepsis-first project-group layer, all while
  preserving source-native ICD codes/text and never dropping rows.
- `CONDITION_MAPPING_SPECS` and `CONDITION_MAPPING_VERSION` in `pipeline/config.py`
  for optional `Dataset/mappings/conditions/` reference files.
- `scripts/build_condition_mappings.py` to inventory distinct diagnosis concepts
  and emit aggregate, review-ready mapping templates (no fabricated mappings).
- `scripts/fetch_condition_reference_files.py` to download authoritative public
  AHRQ HCUP CCSR (ICD-10-CM) and single-level CCS (ICD-9-CM) plus CDC/NCHS
  ICD-9→ICD-10 GEM sources, and derive `icd_chapters.csv` from published chapter
  ranges, writing `icd10_ccsr.csv`, `icd9_ccs.csv`, `icd9_to_icd10_gem.csv`, and
  `icd_chapters.csv` validated against `CONDITION_MAPPING_SPECS`
  (`reports/condition_reference_build_report.json`).
- `tests/test_condition_reference_fetch.py` synthetic offline coverage for the
  CCSR/CCS/GEM parsers, ICD chapter derivation, and spec-column contract.
- `scripts/calculco/harmonize.sh` OAR job script for full harmonization on
  Calculco compute nodes (8 cores, 24h walltime, CPU-only).
- `reports/condition_normalization_coverage.json` and
  `reports/eicu_diagnosis_text_mapping_review.csv` aggregate reports, plus
  `condition_rollup_coverage` in `harmonization_coverage.json` and
  `condition_mapping_resources` in `harmonization_manifest.json`.
- `Documentation/ConditionNormalization.md` freezing the condition contract and
  resolving the shared-condition-vocabulary open decision for this stage.
- `tests/test_condition_normalization.py` and
  `tests/test_condition_mapping_builder.py` synthetic coverage.
- `pipeline/runtime_env.py` with `uv run python -m pipeline.runtime_env` for
  dynamic local vs Calculco layout detection.
- `Documentation/Environment.md` and gitignored machine-specific file conventions.
- `Documentation/CalculcoSetup.example.md` and `scripts/calculco/common.local.sh.example`.
- Completed Milestone 5 harmonization outputs for demographics, labs, vitals,
  allergies, interventions, and temporal events, with per-artifact provenance
  and aggregate coverage reporting.
- `notebooks/03_harmonization_and_overlap.ipynb` for aggregate-only
  harmonization report review.

### Changed

- Replaced hard-coded Calculco paths in committed docs with detection workflow and
  gitignored `*.local.md` / `.env.calculco` / `common.local.sh` patterns.
- Slimmed `Documentation/CalculcoSetup.md` to generic platform notes; preserved
  account details in gitignored `Documentation/CalculcoSetup.local.md` on server.
- `scripts/calculco/common.sh` requires `DATASET_ROOT` from gitignored env files.
- Harmonization and medication-mapping official reports now avoid raw concept
  samples and keep unmapped diagnostics aggregate-only.
- Documented CalcULCO 2026 platform migration (ritchie front-end, chimay nodes,
  OAR property updates, summer outage) in `Documentation/CalculcoSetup.md`.
- `pipeline/runtime_env.py` treats `ritchie` hostnames as Calculco split layout.
- OAR extraction scripts request `gpudevice='-1'` for CPU-only scheduling.

### Fixed

- Harmonization no longer gets SIGKILLed (exit 137) at the eICU `vitals` step.
  The in-memory DuckDB connections in `pipeline/harmonize.py` now disable
  `preserve_insertion_order`, set an explicit spill `temp_directory` (DuckDB
  ignores the OS `TMPDIR`), and honor optional `memory_limit`/`threads` bounds,
  so the large `vital_periodic` `COPY … TO parquet` union streams and spills to
  disk instead of buffering the whole ordered result and being OOM-killed by the
  OAR cgroup before DuckDB's own limit engages. Settings resolve from
  `DUCKDB_TEMP_DIR`/`DUCKDB_MEMORY_LIMIT`/`DUCKDB_THREADS` (see
  `pipeline/config.py`) or `--duckdb-*` CLI flags.
- `scripts/calculco/common.sh` now selects the first writable scratch base
  (`WORK_SCRATCH` → `TMPDIR` → `/tmp`) and always exports a node-local `TMPDIR`
  and `DUCKDB_TEMP_DIR`, instead of aborting when `WORK_SCRATCH`
  (e.g. `/workdir/<lab>` on ritchie/chimay) is not writable.
- `scripts/calculco/harmonize.sh` bounds `DUCKDB_THREADS`/`DUCKDB_MEMORY_LIMIT`
  to the OAR core allocation so DuckDB spills gracefully within the cgroup.

### Added (environment migration, prior)

- Calculco-first execution guidance in `AGENTS.md`, `AGENT-MEMORY.md`,
  `.cursor/rules/core.mdc`, `.codex/config.toml`, and agent skills.
- Verification tiers (lightweight login-node vs OAR HPC) in `AGENTS.md` and
  `TESTING.md`.
- Expanded `.cursorignore`, `.cursorindexingignore`, and `.gitignore` coverage
  for reports, OAR logs, derived outputs, and caches.

### Changed

- Documented Calculco as the primary runtime in `README.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_REVIEW.md`, and `CODING-STANDARDS.md`.
- Updated Codex clinical-research reviewer agent for split-storage and ignore
  rules.
- Clarified that licensed data resolves via `DATASET_ROOT` on protected NFS, not
  only a colocated `Dataset/` directory.

### Added (prior milestones)

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
- `.env.example` for documented runtime path overrides.
- `.gitattributes` to normalize text files to LF across platforms.
- Focused tests for environment-aware path resolution in `tests/test_config.py`.
- Initial Milestone 5 extraction infrastructure:
  `pipeline.extract_utils`, `pipeline.mimic_extract`, and
  `pipeline.eicu_extract` with cohort-filtered DuckDB extraction, report-driven
  quality/integrity gates, local ignored Parquet outputs, and aggregate
  manifests.
- Initial `pipeline.harmonize` CLI for source-tagged cohort, condition, and
  RxNorm/ATC-mapped medication harmonization with mapping-resource validation,
  aggregate coverage, and aggregate unmapped reports.
- Focused synthetic extraction and harmonization tests in
  `tests/test_extraction_harmonize.py`.
- Calculco OAR job scripts under `scripts/calculco/` for report-gated MIMIC-IV
  and eICU source extraction with shared scratch-path setup.

### Changed

- Made `pipeline/config.py` resolve `PROJECT_HOME`, `DATASET_ROOT`,
  `DATA_PROTECTED`, and `REPORTS_ROOT` from the environment for split local/HPC
  layouts.
- Limited pipeline builders to creating their configured output directories
  instead of globally creating all default processed-data directories.
- Replaced Windows-only PowerShell command fences with portable `bash` examples
  across README, workflows, testing, contributing, and roadmap docs.
- Updated Calculco setup guidance with `DATASET_ROOT`, portable transfer
  examples, and current completion status.
- Updated roadmap, architecture, workflows, testing docs, and agent memory for
  the initial extraction/harmonization implementation slice.
- Rebuilt the README to reflect the actual data-foundation stage.
- Expanded `AGENTS.md` with project-specific data, clinical, verification, and
  review rules.
- Clarified that the Transformer-GNN recommender and grounded explanation
  system are target architecture, not completed clinical software.

### Security

- Explicitly excluded licensed datasets, secrets, model artifacts, and
  generated reports from version control and agent indexing.
