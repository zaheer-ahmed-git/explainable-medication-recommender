# Changelog

All notable repository changes are recorded here. Dates use ISO 8601.

## [Unreleased]

### Changed

- Neural Transformer gap-recovery v3 (`phase8-p0-neural-transformer-v3`) after
  protected v2 job 28374 nearly tied Stage 1 NDCG but missed the +0.005 lift and
  failed the MRR guard: join Stage-1-matched train-fit graph tabular features
  (support threshold 5; context + direct summaries that late fusion carries)
  into group caches; residual numeric MLP with feature dropout; candidate-side
  feature MLP; condition-gated dual-path scorer; catalog-primary-positive
  listwise term for MRR; EMA weight selection; lower peak LR (`5e-5`), higher
  weight decay / dropout, aux-BCE `0.05`, patience 2 with min-delta. Requires
  rerunning `prepare` before `train`. GNN message passing and the Stage 1 NDCG
  gate bar remain unchanged.

- Neural Transformer gap-recovery upgrade
  (`phase8-p0-neural-transformer-v2`): numeric MLP stay encoder, learned
  positional encodings, dual-path candidate scorer (MLP + scaled
  context–candidate dot product), train-only global and condition×candidate
  log-odds priors plus `log1p(candidate_rank)` in group caches, AdamW
  warmup+cosine schedule (peak LR `1e-4`, weight decay `1e-3`, dropout `0.2`,
  patience 5), and per-batch aux-BCE `pos_weight` with weight `0.1`. Requires
  rerunning `prepare` before `train`. Synthetic tests cover priors, dual-path
  scoring, and the LR schedule. GNN/fusion remain out of scope; the Stage 1
  NDCG gate bar is unchanged.

### Fixed

- Implemented the versioned GNN P1 representation/modeling contract: explicit
  stay-query nodes and relations; fold-safe numeric value, statistical
  deviation, trend, recency and time-bin attributes; learned relation gates and
  relation dropout; rank-only and dense lab/vital controls; patient-grouped OOF
  temperature fitting; one-fragment cache partitions; effective-group gradient
  accumulation; allocation-local integrity attestations; and independently
  schedulable `(variant, fold)`, selection, and refit commands. This changes the
  GNN feature/relation versions and therefore requires a fresh protected
  `prepare`; no performance claim is made.

- GNN training no longer materializes a dense `E×H×H` relation-weight tensor.
  Weighted source states are accumulated by relation before each transform,
  preserving forward/gradient behavior while removing the allocation that
  requested about 5.5 GiB in protected job 12214. Graph batches now have
  independent group/node/edge ceilings; A100 jobs default explicitly to BF16;
  and aggregate CUDA/progress heartbeats make long epochs observable. Exact
  cross-fit verification hashes each file once per preflight instead of
  repeatedly traversing fold and global trees. Atomic epoch state and reusable
  completed-fold checkpoints are stored outside immutable cross-fit caches and
  accepted only against matching configuration and artifact locks. Calculco
  submissions now run one development stage per OAR job. Job 12214 was
  cancelled; a protected-scale validation run and all performance claims are
  still pending.

- GNN and residual-fusion mixed-precision optimization now treats non-finite
  scaled gradients as recoverable AMP overflow: it skips the unsafe update,
  backs off the loss scale, and retries the same batch up to a bounded limit.
  Persistent overflow and any full-precision non-finite gradient still fail
  closed with model-parameter diagnostics. Epoch history records optimizer
  steps, overflow retries, maximum pre-clip gradient norm, and minimum loss
  scale. The CLI and Calculco wrapper expose an explicit
  `GNN_MIXED_PRECISION=0` fallback while preserving mixed precision by default.
  This addresses the optimization failure in protected job 10962; successful
  protected training remains pending.

- GNN graph-shard loading now accepts every deterministically ordered Parquet
  fragment in one logical Hive split/shard partition, matching DuckDB's
  multi-file `COPY ... PARTITION_BY` output while retaining one logical shard
  as the memory boundary. This fixes the pre-training loader failure observed
  in protected development job 8825 without requiring another `prepare` run.

- Neural `prepare` no longer dies on DuckDB `STDDEV_SAMP is out of range` when
  train stay features contain extreme finite values (job 28346). Mean/std now
  exclude non-finite values and magnitudes above `1e100`, map those cells to the
  normalized mean in the caches, and record aggregate exclusion counts. The OAR
  neural wrapper now uses `/nodes=1/gpu=1` with `#OAR -p gpudevice<>'-1'` (no
  extra quotes; quoted `-p` breaks `oarsub -S`) so jobs do not land on CPU-only
  nodes (28346/28347 on chimay01).

- Updated Stage 2 neural comparison baseline and status docs after Stage 1 gate
  recovery froze `xgboost_rank_ndcg_oof_late_fusion` with
  `neural_training_authorized=true` (validation NDCG@10 ≈ `0.394607`). The
  Transformer now gates against that Stage 1 winner (+0.005 NDCG@10, MRR/Hit
  guards) using gate-recovery `baseline_scores.parquet`, not Milestone 8B
  `xgboost_frozen_reference` (`0.374899`). Fail decision is
  `retain_structured_recovery_baseline`. Default reference paths and OAR
  `--reference-scores` follow the Stage 1 package; docs
  (`TrainingPlan`, roadmap, `AGENT-MEMORY`, `WORKFLOWS`, detailed plan) no
  longer claim neural work is unauthorized.

- Made `pipeline.gate_recovery` development mode memory-safe after two OAR
  runs (28134, 28215) were OOM-killed (exit 137) during full-universe fold
  scoring and pandas OOF fusion. Configuration screening now runs on the
  deterministic train sample only, while the locked finalist, binary reference
  OOF, and validation gate still use the full candidate universe. Finalist and
  reference out-of-fold scores stream to narrow Parquet one fold at a time, and
  the fusion-weight search runs in DuckDB over that Parquet instead of
  concatenating and merging the universe in pandas. Added stdlib progress and
  peak-RSS logging, a contract-keyed screening checkpoint for resume, and a
  filter for the repeated all-null median-imputation warning. The CPU worker
  `scripts/calculco/phase8_p0_gate_recovery.sh` now defaults to
  `DUCKDB_THREADS=8`, `DUCKDB_MEMORY_LIMIT=24GB`, sets `MALLOC_ARENA_MAX=2` and
  `PYTHONUNBUFFERED=1`, and warns when DuckDB spill falls back to `/tmp`.
  Screening selection is now sample-based; the neural-readiness gate metric and
  its fail-closed contract are unchanged.

### Added

- Phase D GNN and frozen-Transformer fusion workflow in
  `pipeline.gnn_training`: a lazy five-stage CLI; exact immutable
  contracts; full-train refit and patient-fold-excluded graph caches; frozen
  Transformer representation extraction; a native PyTorch R-GCN-style
  medication ranker with four pre-registered relation ablations; standalone
  selection/refit/calibration/scoring; late and zero-initialized residual
  fusion; canonical DuckDB scoring; and atomic one-shot final gates. Added
  focused synthetic contract, leakage, temporal-cutoff, batching, model,
  scoring, non-finite-value, final-claim, and end-to-end tests. Added separate
  capacity-gated CPU prepare and GPU train/score Calculco wrappers. Protected
  preparation later completed; successful training and metrics remain pending
  after loader failure job 8825 and mixed-precision optimization failure job
  10962.

- Stage 2 conditional neural Transformer patient/context training pipeline:
  `pipeline.neural_training` implements a `prepare`/`train`/`score` CLI with
  `development` and `final` modes. `prepare` builds train-fit event/condition/
  candidate/categorical vocabularies (reserved `PAD=0`/`UNK=1`), train-only
  numeric and event-value normalization, a persisted `feature_layout.json`, and
  memory-bounded hash-sharded DuckDB caches. `train` fits a Transformer event
  encoder plus static-context encoder and candidate scorer with a multi-positive
  listwise softmax and auxiliary BCE objective, early stopping on validation
  NDCG@10, gradient clipping, optional mixed precision, checkpointing, and
  post-hoc temperature calibration. `score` writes canonical
  `baseline_scores.parquet` rows, computes authoritative DuckDB metrics against
  the Stage 1 structured recovery baseline, and records a neural gate/selection
  report. Every
  stage is fail-closed behind the structured recovery gate
  (`neural_training_authorized`) and the training-contract lock. PyTorch is added
  as an optional `neural` dependency group (imported lazily) and is not synced by
  default. Added synthetic torch-guarded tests for data prep, ranking metrics,
  dataset collation, the model/losses, the preflight/gate, and an end-to-end
  smoke test, plus a configurable GPU OAR wrapper
  (`scripts/calculco/phase8_p0_neural_training.sh`) and job env template.

- Gate-first Phase 8 P0 Stage 1 implementation: `pipeline.training_contract`
  locks pinned versions, aggregate manifest hashes, artifact metadata/schemas,
  row counts, fit scope, split integrity, and temporal-label boundaries;
  `pipeline.gate_recovery` performs patient-fold rank-aware XGBoost screening,
  train-OOF fusion selection, one-shot validation gating, and fail-closed final
  scoring using the existing baseline score schema and metrics. Added focused
  synthetic tests and CPU-only Calculco submit/worker wrappers. PyTorch was kept
  out of this Stage 1 change; the Stage 2 neural branch (see the entry above) was
  added separately and stays fail-closed behind this gate.

- CodexPLAN Step 10 graph/hybrid readiness review on the Phase 8 P0 stack:
  `Documentation/CodexPLANStep10GraphHybridReadiness.md` and aggregate
  `reports/codexplan_step10_graph_hybrid_readiness.json`. Graph structure gate
  passes (`pass_for_graph_ablation`); Milestone 8B hybrid lift fails, so frozen
  tabular XGBoost remains the reference and neural Transformer-GNN training is
  not authorized yet.

- `scripts/calculco/submit_phase8_p0_model_ready.sh` wrapper and
  `phase8_p0_model_ready_job.env` (gitignored, with a committed `.example`) so
  OAR jobs receive `PHASE8_P0_START_AT` and the subgraph memory knobs. OAR `-S`
  jobs run in a clean environment and ignore login-shell exports, which caused a
  resume-at-subgraphs attempt to silently rerun from training. `phase8_p0_model_ready.sh`
  now sources the job env file, matching the Milestone 7/8B submit pattern.

- Complete CodexPLAN Step 9 Phase 8 P0 package implementation:
  `pipeline.patient_subgraphs` materializes normalized per-ranking-group node,
  edge, candidate, and index artifacts from train-fit graph edges;
  `pipeline.model_ready_package` writes train-derived vocabularies, a
  schema-only data dictionary, and an aggregate completion manifest with
  explicit eICU evaluability gates. Downstream builders now infer and validate
  feature provenance from their inputs, and the training builder writes a
  model-ready `cohort_stays` artifact. The Calculco chain now runs both
  RxNorm-first and ATC-3-first label builds before graph/package assembly.
- Aggregate-only Phase 4-9 meeting visualization pack:
  `visualization.phase4_to_9` reads completed report manifests, writes ignored
  charts under `visualization/figures/`, and creates
  `visualization/meeting_figure_pack.md` / `.json` for briefing current feature,
  preprocessing, baseline, graph-readiness, and graph-ablation status without
  patient-level rows or clinical recommendation claims.
- Phase 8 P0 feature-ablation path:
  `pipeline.features --feature-set phase8_p0` now writes isolated
  `temporal-features-v2` artifacts with train-only condition presence columns,
  core lab/vital trend summaries, explicit missingness indicators, and
  aggregate-only feature-family/OOV manifest fields. Added
  `scripts/calculco/phase8_p0_features.sh`,
  `scripts/calculco/phase8_p0_model_ready.sh`,
  `pipeline.feature_gate_review`, and synthetic coverage for train-only
  vocabularies, trend boundaries, downstream feature pickup, and gate-review
  reject behavior.
- CodexPLAN §7 preprocessing completion bridge: `pipeline.preprocessing` now
  fits train-only imputation, scaling, encoding, and categorical vocabulary
  artifacts from MIMIC train rows, writes ignored preprocessing artifacts under
  `Dataset/processed/training/preprocessing/`, and emits aggregate-only
  `reports/preprocessing_manifest.json`.
- Harmonization/training integrity coverage for preprocessing: eICU cancelled
  medication orders are filtered before labels, repeated domain events are
  deduplicated deterministically, aggregate cleanup counts are reported, and
  modeling input join-integrity failures stop training-table construction.
- `Documentation/HybridModelFeatureStrategy.md`: canonical planning reference
  for post-8B Transformer/GNN feature families, branch boundaries, selection
  gates aligned with Milestone 7/8B metrics, and explicit implemented-vs-planned
  status without changing current pipeline scope. Molecular drug graphs are
  out of scope for this project direction.
- Milestone 8B graph-aware ablation tooling:
  `pipeline.graph_ablation` builds graph-derived candidate features from the
  train-fit Milestone 8 graph, compares graph-only XGBoost,
  graph-augmented XGBoost, validation-weighted late fusion, and a simple
  ensemble against the frozen Milestone 7 XGBoost reference, and writes
  aggregate-only `reports/milestone8b_graph_feature_manifest.json`,
  `reports/milestone8b_ablation_evaluation.json`, and
  `reports/milestone8b_frozen_selection.json`.
- `scripts/calculco/graph_ablation.sh`,
  `scripts/calculco/submit_graph_ablation.sh`,
  `Documentation/Milestone8B.md`, and `tests/test_graph_ablation.py` for
  CPU-only OAR execution, graph-ablation workflow documentation, and synthetic
  train-fit graph feature/fusion/final-gating/report-safety coverage.
- Milestone 8 graph-readiness tooling:
  `pipeline.graph_suitability` builds train-only concept-level graph edges under
  ignored `Dataset/processed/graph/milestone8/`, writes aggregate-only
  `reports/milestone8_graph_schema.json`,
  `reports/milestone8_graph_suitability.json`, and
  `reports/milestone8_ablation_plan.json`, and records graph gate status
  without training the Transformer-GNN model.
- `scripts/calculco/graph_suitability.sh`,
  `Documentation/Milestone8.md`, `notebooks/04_graph_suitability.ipynb`, and
  `tests/test_graph_suitability.py` for OAR execution, aggregate-only review,
  and synthetic train-only/leakage/cold-start coverage.
- Coverage-bottleneck implementation support:
  `scripts/build_condition_mappings.py --write-curated-sepsis` now merges the
  approved A1/B3 sepsis ICD/text policy into active local condition mapping
  CSVs, `pipeline.harmonize` supports `project_condition_groups.csv`
  `match_type=icd_prefix`, and `pipeline.build_training_table` adds
  `--candidate-token-strategy atc3_or_rxnorm` for ATC-3-first
  coverage-sensitivity candidate catalogs. Added synthetic tests for each path.
- Protected-data Milestone 7 validation summary and frozen-selection records:
  `reports/milestone7_validation_summary.json` (job 2084 development metrics)
  and `reports/milestone7_frozen_selection.json` (headline baseline `xgboost`).
- `scripts/calculco/submit_evaluate_baselines.sh` and
  `scripts/calculco/milestone7_job.env.example` so OAR jobs source durable
  Milestone 7 env controls via gitignored `milestone7_job.env`.
  (`SGDClassifier`) and XGBoost models trained on MIMIC train positives plus a
  deterministic 5:1 weak-negative sample, with local ignored model artifacts and
  batched scoring integrated into `pipeline.evaluate_baselines`.
- Extended `tests/test_milestone7_baselines.py` with learned-baseline,
  row-level null-metric, and Milestone 6 integration coverage.
- Initial Milestone 7 baseline evaluation scaffold:
  `pipeline.evaluate_baselines` writes aggregate-only
  `reports/milestone7_coverage_report.json` and
  `reports/milestone7_baseline_evaluation.json`, with local row-level scores
  under ignored `Dataset/processed/evaluation/milestone7/`. The first slice
  implements coverage/evaluability checks, deterministic random, global
  popularity, and condition-popularity baselines, ranking/calibration metrics,
  and final/test blocking unless `--mode final --frozen-selection` is explicit.
- `EVALUATION_ROOT`, `MILESTONE7_EVALUATION_ROOT`, `BASELINE_VERSION`, and
  `EVALUATION_VERSION` in `pipeline/config.py`, plus the Calculco OAR wrapper
  `scripts/calculco/evaluate_baselines.sh`.
- `tests/test_milestone7_baselines.py` covering synthetic metric behavior,
  train-only popularity fitting, deterministic random scores, report safety,
  and final-mode gating.
- `Documentation/Milestone6MaterializationReview.md` and aggregate
  `reports/milestone6_materialization_review.json` documenting protected-data
  Milestone 6 completion (OAR jobs 830/1055), exit gates, and open items before
  Milestone 7.
- Milestone 6 temporal feature and observed-label builders:
  `pipeline.features` writes `cohort_decision_times.parquet`,
  `patient_stay_features.parquet`, and `event_sequences.parquet`, and
  `pipeline.build_training_table` writes `split_manifest.parquet`,
  `candidate_catalog.parquet`, and `patient_condition_medication.parquet` under
  ignored `Dataset/processed/` subdirectories with aggregate-only manifests.
- `FEATURES_ROOT`, `TRAINING_ROOT`, `FEATURE_VERSION`, `LABEL_VERSION`, and
  `SPLIT_VERSION` in `pipeline/config.py`.
- `Documentation/Milestone6FeatureLabelDictionary.md` documenting the temporal
  contract, schemas, train-only candidate rules, censoring, and
  observational-label caveats.
- OAR wrappers `scripts/calculco/profile_tables.sh` (full source-table
  re-profile), `scripts/calculco/features.sh`, `scripts/calculco/build_training_table.sh`,
  and `scripts/calculco/milestone6.sh` (features + training-table chain) for
  protected-data Milestone 6 materialization on Calculco.
- `Documentation/SepsisCohortAndIndexConditionPolicy.md` recording the approved
  coded sepsis sub-cohort definition (A1; Sepsis-3 deferred) and the B1 -> B3
  index-condition/ranking-group policy, with follow-up implementation steps.
- MIMIC `chartevents` charted-vital extraction: a gated `mimic_chartevents`
  spec in `pipeline.mimic_extract` restricted to curated core-vital itemids
  (`MIMIC_CHARTEVENTS_VITAL_ITEMIDS` in `pipeline/config.py`) via a new optional
  `ExtractionTableSpec.source_row_filter`, plus a MIMIC chartevents branch in
  `pipeline.harmonize` `vital_queries` that maps those itemids to the shared
  `normalized_vital_token` vocabulary. Adds synthetic coverage in
  `tests/test_extraction_harmonize.py`.
- `tests/test_features.py` and `tests/test_build_training_table.py` covering
  MIMIC timestamps, eICU offsets, 24h/48h boundaries, censoring, deterministic
  splits, train-only candidates, repeated medications, out-of-catalog positives,
  and aggregate-only manifests.
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

- Milestone 7 metric aggregation now runs one `(baseline_name, source, split)`
  slice at a time. Each row-level and ranking query sorts a single window
  partition instead of the whole combined score table, keeping final-mode DuckDB
  memory and temp spill bounded; per-slice results are unioned in Python and are
  numerically identical to the previous whole-table query.
- Learned Milestone 7 training now samples weak negatives on narrow
  `patient_condition_medication` rows with a deterministic per-condition hash
  threshold before joining the wide `patient_stay_features` table. XGBoost
  matrix assembly now uses sparse batch stacking instead of concatenating all
  training pandas frames.
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

- Phase 8 P0 patient-subgraph edge join no longer exhausts DuckDB's spill
  directory (`failed to offload data block ... (X GiB/X GiB used)`). Root cause:
  popular concept nodes make the naive `graph_edges`-to-`dst`-membership join fan
  out across nearly all ~916k subgraphs (e.g. a common candidate medication is a
  member of 915k subgraphs), producing billions of intermediate rows. The
  per-relation expansion is now forced to start from the small `src` side (one
  query condition per subgraph; ~50 candidate medications) behind a
  `MATERIALIZED` optimization barrier, so the `subgraph_key` correlation is
  applied before touching popular `dst` nodes. The self-joins also carry only
  integer keys (`subgraph_key`, `src_node_index`, `dst_node_index`, `edge_id`)
  and attach wide string/provenance columns once at the end via an `edge_id`
  join back to the small train-fit edge relation. Verified on the real 94.8M-node
  data: edge shard 0/64 (which killed jobs 7563/7617/7639/7660) now completes in
  ~46s under a 12 GiB spill cap. Also fixes the 32-shard workaround that OAR
  killed after 3h+ from repeated full node-batch re-scans.
- DuckDB spill capacity is now explicitly configurable via
  `DUCKDB_MAX_TEMP_DIR_SIZE` (`pipeline.config.resolve_duckdb_max_temp_dir_size`,
  `configure_duckdb_connection`, and `pipeline.patient_subgraphs
  --duckdb-max-temp-dir-size`), so operators can raise the cap when
  `DUCKDB_TEMP_DIR` points at a larger volume instead of inheriting the small
  auto-detected default. Wired through `phase8_p0_model_ready.sh`,
  `submit_phase8_p0_model_ready.sh`, and the job env example.
- Phase 8 P0 patient-subgraph spill exhaustion now has bounded
  source-qualified stay-hash node materialization plus independently sharded,
  integer-encoded, relation-specific edge and candidate joins. The OAR chain
  exposes `SUBGRAPH_JOIN_SHARDS`, caps edge assembly at one DuckDB thread, can
  resume at the subgraph stage, and prefers node-local `/scratch` over `/tmp`.
- Harmonization lab/vital OOMs now have a bounded materialization path:
  `pipeline.harmonize` writes large lab and vital domains as smaller
  source-query/hash-batched parts, combines them into the canonical Parquet
  artifacts only after successful part writes, records batch metadata in the
  manifest, and exposes `--domain-materialization-batches`.
- Milestone 6 `patient_stay_features` now has the same bounded-memory pattern:
  `pipeline.features` writes stay-hash-batched feature parts before combining
  the canonical Parquet artifact, records batch metadata in the manifest, and
  exposes `--stay-feature-batches`.
- `scripts/calculco/features.sh` and `scripts/calculco/milestone6.sh` now use
  safer default DuckDB settings for the observed Calculco cgroup
  (`DUCKDB_THREADS=4`, `DUCKDB_MEMORY_LIMIT=10GB`) and pass
  `STAY_FEATURE_BATCHES` to the feature CLI.
- `scripts/calculco/harmonize.sh` now uses safer default DuckDB settings for
  the observed Calculco cgroup (`DUCKDB_THREADS=4`,
  `DUCKDB_MEMORY_LIMIT=10GB`) and passes `HARMONIZE_DOMAIN_BATCHES` to the
  harmonization CLI.
- Milestone 6 `event_sequences` now avoids one global `ROW_NUMBER()` over the
  full pre-decision temporal-event set. `pipeline.features` stages the reduced
  pre-decision events once, windows them in configurable stay-hash batches
  (`--event-sequence-batches`, default 8), and combines the parts into the
  canonical single `event_sequences.parquet`; the Calculco feature and
  milestone wrappers also honor `EVENT_SEQUENCE_BATCHES`.
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
