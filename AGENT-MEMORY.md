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
- CodexPLAN Step 10 graph/hybrid readiness (2026-07-18) is reviewed on the
  Phase 8 P0 stack in
  `Documentation/CodexPLANStep10GraphHybridReadiness.md` and
  `reports/codexplan_step10_graph_hybrid_readiness.json`: structure gate
  `pass_for_graph_ablation`, Milestone 8B hybrid lift failed at that review
  (frozen XGBoost retained). Subsequent Stage 1 gate recovery superseded that
  authorization block.
- Gate-first Phase 8 P0 Stage 1 is implemented in `pipeline.training_contract`
  and `pipeline.gate_recovery`. Protected development completed: frozen
  `xgboost_rank_ndcg_oof_late_fusion` (validation NDCG@10 ≈ `0.394607`) with
  `neural_training_authorized=true`. The contract lock is aggregate only;
  structured selection used MIMIC-train patient folds. Optional Stage 1 final
  MIMIC test scoring remains available when requested.
- Stage 2 neural training is implemented in the `pipeline.neural_training`
  subpackage (`config`, `contract`, `data`, `dataset`, `model`, `losses`,
  `metrics`, `train`, `score`, `__main__`) as a `prepare`/`train`/`score` CLI
  with `development` and `final` modes. It trains and freezes the
  Transformer patient/context branch. PyTorch is an optional `neural`
  dependency group in `pyproject.toml`,
  imported lazily so `config`, `contract`, `data`, `metrics`, and CLI parsing
  work without torch. Every stage is fail-closed: it verifies
  `reports/phase8_p0_training_contract_lock.json` and requires
  `neural_training_authorized=true` in
  `reports/phase8_p0_gate_recovery_selection.json` (bypass only via
  `require_neural_gate=False` for synthetic smoke tests). The neural gate bar
  is the Stage 1 winner (default scores
  `.../evaluation/gate_recovery/baseline_scores.parquet`, baseline name
  `xgboost_rank_ndcg_oof_late_fusion`), not Milestone 8B `0.374899`; fail
  decision is `retain_structured_recovery_baseline`. Caches, vocabularies,
  checkpoints, predictions, and row-level scores stay ignored under
  `Dataset/processed/phase8_p0/neural/`; aggregate reports are
  `reports/phase8_p0_neural_prepare_manifest.json`,
  `reports/phase8_p0_neural_training_evaluation.json`,
  `reports/phase8_p0_neural_score_evaluation.json`, and
  `reports/phase8_p0_neural_training_selection.json`. Reserved vocabulary tokens
  are `PAD=0`/`UNK=1` (indices offset by 2); numeric/event-value normalization and
  candidate priors are train-only; the primary seed is 20260617. Experiment
  `phase8-p0-neural-transformer-v2` added a numeric MLP, learned PE, dual-path
  scorer, train-fit priors/`log1p(candidate_rank)`, and warmup+cosine AdamW;
  protected v2 (job 28374) reached NDCG@10 ≈ 0.3955 but failed the +0.005 /
  MRR gate. `phase8-p0-neural-transformer-v3` adds Stage-1-matched train-fit
  graph tabular side features (support 5), residual numeric MLP with feature
  dropout, candidate-side MLP, condition-gated scorer, primary-positive
  listwise for MRR, EMA selection, and stronger regularization; rerun
  `prepare` after upgrading. The GPU OAR wrapper is
  `scripts/calculco/phase8_p0_neural_training.sh` (installs the `neural` group).
  The first protected development run (v1) completed but failed the neural gate
  (best epoch 0, NDCG@10 ≈ 0.3747 vs Stage 1 ≈ 0.3946).
- Phase D is implemented in `pipeline.gnn_training`. Its five-stage CLI
  prepares a full-train refit cache plus five patient-fold-excluded selection
  caches, extracts immutable Transformer contexts/logits, selects four native
  PyTorch R-GCN-style ablations, refits and qualifies a standalone relation
  branch, and trains/qualifies late and zero-initialized residual fusion. Exact
  artifact hashes, canonical candidate reconciliation, non-finite-value
  rejection, and atomic one-shot final markers fail closed. The full-train
  graph is never selection-eligible. The Transformer cache is a full-train
  refit rather than Transformer OOF, so late alpha is documented as train
  meta-fitting over GNN OOF plus a fixed Transformer covariate; promotion uses
  the separate validation gate. Focused synthetic tests pass, including the
  complete workflow. Protected preparation and cross-fit caches completed.
  Development job 8825 failed before optimization because edge partitions can
  contain multiple DuckDB Parquet fragments; the loader now assembles all
  fragments deterministically. A replacement training/scoring run and all
  protected GNN/fusion metrics remain pending.
  CPU prepare and GPU train/score wrappers are
  `scripts/calculco/phase8_p0_gnn_prepare.sh` and
  `scripts/calculco/phase8_p0_gnn_training.sh`; prepare requires
  `WORK_SCRATCH` and a reviewed `GNN_CROSSFIT_MIN_FREE_GIB`.
- `pipeline.gate_recovery` development mode is memory-bounded after OAR jobs
  28134/28215 were OOM-killed (exit 137): the earlier code scored every
  screening fold against the full ~25.7M-row universe and built both candidate
  and reference OOF plus the fusion search in pandas. Now screening runs on the
  deterministic train sample; the locked finalist, reference OOF, and validation
  gate use the full universe streamed to narrow Parquet one fold at a time; and
  the fusion-weight search runs in DuckDB. It logs stage progress and peak RSS,
  writes a contract-keyed `screening_checkpoint.json` for resume, and the worker
  defaults to `DUCKDB_THREADS=8` / `DUCKDB_MEMORY_LIMIT=24GB` (DuckDB's ceiling
  does not cover pandas/sklearn/XGBoost, so keep it below the node budget) with
  `MALLOC_ARENA_MAX=2`. Screening selection is sample-based; the gate metric and
  fail-closed contract are unchanged.
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
  `graph_ablation.sh`, `submit_graph_ablation.sh`, the `milestone6.sh`
  chain, the GPU `phase8_p0_neural_training.sh` Stage 2 wrapper, and
  the Phase D GNN CPU-prepare/GPU-training wrappers.
- `pipeline.profile_tables` rewrites the entire `reports/quality_profile.json`;
  re-profile all tables (not a `--table` subset) so extraction gate entries are
  preserved.
- Sepsis sub-cohort extraction and detailed EDA notebooks remain unimplemented.
  Transformer, GNN relation, and fusion workflow code exists, but Phase D has
  not been run on protected data. A reproducible sepsis definition and
  index-condition policy are
  proposed for approval in
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
- DuckDB `failed to offload data block (.../12.2 GiB used)` on Calculco means
  the node-local spill allowance was exhausted. Patient-subgraph construction
  uses stay batches for node construction and separate integer-membership join
  shards for edges/candidates. Tune `SUBGRAPH_JOIN_SHARDS` upward for join
  failures; tune `SUBGRAPH_BATCHES` only for node failures. OAR scratch selection
  prefers `WORK_SCRATCH`, then `/scratch`, before `/tmp`.
- Neural GPU OAR: `/nodes=1/gpu=1` alone can still land on CPU hosts (chimay01).
  Require `#OAR -p gpudevice<>'-1'` (mirror of CPU `gpudevice='-1'`). Do not wrap
  that `-p` expression in extra double quotes inside `#OAR` lines — `oarsub -S`
  then fails immediately with "There are not enough resources" / `OAR_JOB_ID=-5`.
  CLI `-p "…"` is fine; the `#OAR` script form is not.

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
