# Agent Memory

This file contains stable, versioned project facts. It is not a substitute for
task context, source-code inspection, or local agent memory.

## Stable Facts

- Development runs on **ULCO Calculco**. Export `PROJECT_HOME`, `DATASET_ROOT`,
  and `WORK_SCRATCH` before pipeline work.
- Account-specific Calculco paths belong in gitignored
  `Documentation/CalculcoSetup.local.md` (template:
  `Documentation/CalculcoSetup.example.md`). Generic platform notes:
  `Documentation/CalculcoSetup.md`. Path configuration: `Documentation/Environment.md`.
- The research topic is an explainable conversational medication recommender
  for clinician-facing decision support.
- Recommendation generation and explanation generation are separate modules.
- The target recommender combines Transformer context modeling and
  heterogeneous GNN relation modeling.
- Explanations should combine attribution, graph evidence, rule checks,
  uncertainty, contradiction handling, and provenance.
- The main structured task is ranking medication candidates for a
  patient/stay-condition context.
- MIMIC-IV v3.1, MIMIC-IV-Note v2.2, and eICU-CRD v2.0 are licensed data
  resolved via `DATASET_ROOT` on protected NFS.
- Raw datasets are licensed, de-identified, ignored, and non-redistributable.
- `uv` is the only supported Python environment and dependency tool.
- Python 3.13 is the declared runtime.
- On Calculco, export `PROJECT_HOME`, `DATASET_ROOT`, and
  `WORK_SCRATCH` via `.env.calculco` or `scripts/calculco/common.local.sh`
  (both gitignored). `scripts/calculco/common.sh` requires `DATASET_ROOT`.

## Current Repository State

- The active data-foundation pipeline skeleton and focused tests are present as
  of 2026-06-20; full local cohort-filtered extraction and harmonization runs
  completed by 2026-07-01.
- `pipeline/source_inventory.py` generates metadata-only
  `reports/source_inventory.json`; `reports/` remains ignored.
- `pipeline/cohort.py` generates adult MIMIC-IV/eICU cohort artifacts under
  ignored `Dataset/processed/cohorts/` and aggregate
  `reports/cohort_manifest.json`.
- `pipeline/profile_tables.py` generates aggregate-only
  `reports/quality_profile.json`; the 2026-06-18 default run completed 22 of 24
  configured structured tables and recorded scan failures for MIMIC
  `chartevents` and `inputevents` that are stale relative to corrected local
  files; re-profile after source correction.
- `pipeline/eda_summary.py` synthesizes aggregate inventory, cohort, and
  quality reports into ignored `reports/eda_dataset_understanding.json`,
  `reports/eda_dataset_understanding.md`, and figures under `reports/figures/`.
- `pipeline/source_integrity.py` checks profiling-blocked files against local
  `SHA256SUMS.txt` manifests and gzip integrity. The 2026-06-30 targeted audit
  shows all six previously blocked tables, including MIMIC `chartevents` and
  `inputevents`, passing checksum/gzip gates.
- `pipeline/mimic_extract.py` and `pipeline/eicu_extract.py` provide
  report-gated, cohort-filtered source extraction CLIs. Full local runs
  completed 2026-06-28 (MIMIC 10/11 tables; eICU 12/12); MIMIC `inputevents`
  was skipped due to a stale quality gate; `chartevents` extraction is not yet
  in the CLI.
- `pipeline/harmonize.py` provides harmonization for cohort, demographics,
  conditions, RxNorm/ATC-mapped medications, labs, vitals, allergies,
  interventions, and temporal events. Latest local run completed 2026-07-01.
- Calculco OAR submission scripts for protected-data extraction live in
  `scripts/calculco/`; submit with `oarsub -S` from the login node, not
  interactively on the login node.
- Sepsis sub-cohort extraction, EDA notebooks, feature tables, and models are
  not yet implemented.
- `DepreciatedCode/` contains the ignored synthetic prototype.
- The prototype includes preprocessing, deterministic patient splitting,
  linear and XGBoost ranking, and ranking metrics.
- `Documentation/ResearchDetail.md` is the current research framing.
- `Documentation/OldResearchDetail.md` is historical.
- `Documentation/DataFoundationRoadmap.md` is the implementation roadmap.
- `FinalPosterCDS.pdf` is an architectural research poster, not proof of a
  completed clinical system.

## Known Pitfalls

- The dataset directory is singular: `Dataset/`, not `Datasets/`.
- The legacy directory is spelled `DepreciatedCode/`; preserve the path until a
  deliberate migration.
- Older notes incorrectly state that MIMIC-IV-Note is absent.
- Older README content described active modules that are no longer in the
  working tree.
- MIMIC timestamps are shifted and are not real calendar dates.
- eICU is multi-center while MIMIC-IV is single-center; source differences are
  meaningful, not noise to erase.
- Observed prescriptions are not equivalent to optimal treatment labels.
- Unobserved candidate medications are not guaranteed clinical negatives.
- Outcome and medication-history features can leak future or target
  information.
- DuckDB harmonization can be SIGKILLed (exit 137, empty stderr) at the eICU
  `vitals` step even with free RAM: an in-memory database preserves insertion
  order and does not read the OS `TMPDIR`, so large ordered `COPY … TO parquet`
  unions buffer in memory and exceed the OAR cgroup before DuckDB's own limit
  engages. Always configure connections via `configure_duckdb_connection`
  (`preserve_insertion_order=false`, explicit spill `temp_directory`, bounded
  `memory_limit`/`threads`); tune with `DUCKDB_TEMP_DIR`/`DUCKDB_MEMORY_LIMIT`/
  `DUCKDB_THREADS`.

## Do Not Do

- Do not assume a local laptop or monolithic checkout layout.
- Do not hard-code Calculco NFS paths in source code; use environment variables.
- Do not run heavy pipeline jobs interactively on the Calculco login node.
- Do not commit or quote patient-level data.
- Do not load multi-gigabyte source tables into pandas.
- Do not pool MIMIC and eICU before measuring mapping and cohort compatibility.
- Do not claim clinical validity from synthetic or poster examples.
- Do not revive deleted code based only on stale documentation.
- Do not add dependencies outside `uv`.
