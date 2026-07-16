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
  was skipped due to a stale quality gate. A gated `mimic_chartevents` spec
  (charted vitals, restricted to `MIMIC_CHARTEVENTS_VITAL_ITEMIDS` via
  `ExtractionTableSpec.source_row_filter`) is now in the CLI but, like
  `inputevents`, materializes only after a refreshed quality/integrity profile.
- MIMIC charted vitals reach harmonized `vitals.parquet` through a
  `pipeline.harmonize` chartevents branch that maps the curated itemids to the
  shared `normalized_vital_token` vocabulary; before this, harmonized MIMIC
  vitals were effectively empty (only eICU vital tables were wired).
- `pipeline/harmonize.py` provides harmonization for cohort, demographics,
  conditions, RxNorm/ATC-mapped medications, labs, vitals, allergies,
  interventions, and temporal events. Latest local run completed 2026-07-01.
- `pipeline/features.py` and `pipeline/build_training_table.py` implement the
  initial Milestone 6 temporal feature, patient split, train-only candidate
  catalog, and observed-label ranking-table artifacts with aggregate-only
  manifests and synthetic tests. Protected-data materialization completed on
  Calculco 2026-07-05/06 (OAR jobs 830/1055); see
  `Documentation/Milestone6MaterializationReview.md`.
- `pipeline/evaluate_baselines.py` and `pipeline/learned_baselines.py`
  implement the Milestone 7 P0-P3 scaffold: aggregate coverage/evaluability
  reporting, deterministic random, global-popularity, condition-popularity,
  linear, and XGBoost baselines, aggregate ranking/calibration metrics, and
  final/test gating. Learned baselines sample positives and deterministic weak
  negatives on narrow `patient_condition_medication` rows before joining wide
  stay features to avoid DuckDB window-sort OOMs. Metric aggregation in
  `append_metric_summaries` runs one `(baseline_name, source, split)` slice at a
  time (via `metric_slices` / `metric_slice_predicate`) so window sorts stay
  bounded on the large final-mode score table; per-slice results are identical
  to the old whole-table query. Local row-level scores and
  model artifacts are ignored under
  `Dataset/processed/evaluation/milestone7/`; aggregate reports are
  `reports/milestone7_coverage_report.json`,
  `reports/milestone7_baseline_evaluation.json`,
  `reports/milestone7_validation_summary.json`, and
  `reports/milestone7_frozen_selection.json`. Use
  `scripts/calculco/submit_evaluate_baselines.sh` so `milestone7_job.env` is
  written before `oarsub`.
- `pipeline/graph_suitability.py` implements Milestone 8 graph-readiness:
  train-only concept-level graph edges under
  `Dataset/processed/graph/milestone8/`, aggregate schema/suitability/ablation
  reports under `reports/milestone8_*.json`, and synthetic tests in
  `tests/test_graph_suitability.py`. This is not Transformer-GNN training.
- `pipeline/graph_ablation.py` implements Milestone 8B graph-aware ablations:
  graph-derived candidate features, graph-only XGBoost, graph-augmented
  XGBoost, validation-weighted late fusion, and a simple ensemble against the
  frozen XGBoost reference. Local artifacts are ignored under
  `Dataset/processed/evaluation/milestone8b/`; aggregate reports are
  `reports/milestone8b_graph_feature_manifest.json`,
  `reports/milestone8b_ablation_evaluation.json`, and
  `reports/milestone8b_frozen_selection.json`. This is still not full neural
  Transformer-GNN training or a clinical recommendation system.
- Calculco OAR submission scripts for protected-data work live in
  `scripts/calculco/`; submit with `oarsub -S` from the login node, not
  interactively on the login node. These include extraction (`extract_*.sh`),
  `harmonize.sh`, `profile_tables.sh` (full source-table re-profile),
  `features.sh`, `build_training_table.sh`, `evaluate_baselines.sh`,
  `submit_evaluate_baselines.sh`, `graph_suitability.sh`,
  `graph_ablation.sh`, `submit_graph_ablation.sh`, and the `milestone6.sh`
  chain.
- `pipeline.profile_tables` rewrites the entire `reports/quality_profile.json`;
  re-profile all tables (not a `--table` subset) so extraction gate entries are
  preserved.
- Sepsis sub-cohort extraction, detailed EDA notebooks, graph neural models, and
  hybrid Transformer-GNN training are not yet implemented. A reproducible sepsis definition and
  index-condition policy are proposed for approval in
  `Documentation/SepsisCohortAndIndexConditionPolicy.md`.
- `DepreciatedCode/` contains the ignored synthetic prototype.
- The prototype includes preprocessing, deterministic patient splitting,
  linear and XGBoost ranking, and ranking metrics.
- `Documentation/ResearchDetail.md` is the current research framing.
- `Documentation/OldResearchDetail.md` is historical.
- `Documentation/DataFoundationRoadmap.md` is the implementation roadmap.
- `Documentation/HybridModelFeatureStrategy.md` records planned hybrid
  feature boundaries and selection gates; it does not implement neural models.
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
