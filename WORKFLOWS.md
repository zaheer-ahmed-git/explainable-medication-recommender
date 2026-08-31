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

```bash
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

```bash
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

## Build Phase 4-9 Meeting Visuals

1. Confirm the aggregate manifests through the current work state exist under
   `reports/`, especially Milestone 6, Milestone 7, Milestone 8, and Milestone
   8B reports.
2. Run:

```bash
uv run python -m visualization.phase4_to_9
```

3. Review `visualization/meeting_figure_pack.md` and
   `visualization/figures/`.
4. Keep the generated files ignored; they are aggregate meeting artifacts, not
   source data or tracked research state.
5. Do not convert observed-label, graph-ablation, or coverage-only findings
   into clinical recommendation claims.

## Check Source Integrity

1. Run `uv run python -m pipeline.source_integrity` for profiling-blocked files.
2. Review `reports/source_integrity_failed_tables.json`.
3. Run `uv run python -m pipeline.source_integrity --all-manifest-files` for a
   complete checksum/gzip audit of all files listed in configured
   `SHA256SUMS.txt` manifests.
4. Use `--source mimiciv`, `--source eicu_crd`, or `--source mimiciv_note` to
   split the full audit into smaller source-specific runs.
5. Treat checksum mismatches, truly missing local files, or gzip failures as
   source-integrity blockers. If a configured uncompressed local file exists
   for a manifest `.csv.gz` entry, document it as a source-layout
   reconciliation before using the table downstream.
6. Re-transfer or re-download affected files before extraction or feature
   engineering.
7. Consider CSV parser fallbacks only after checksum and gzip validation pass.

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

Current extraction commands:

```bash
uv run python -m pipeline.mimic_extract
uv run python -m pipeline.eicu_extract
```

Extraction commands depend on `reports/quality_profile.json`,
`reports/source_integrity_failed_tables.json`, and
`Dataset/processed/cohorts/cohort_stays.parquet`. Blocked tables are skipped
with aggregate manifest entries rather than forced through parser or integrity
failures.

## Run Milestone 5 Harmonization

1. Confirm source-specific extraction manifests are available and aggregate-only.
2. Place reviewed medication mapping files under
   `Dataset/mappings/medications/`:
   `mimic_ndc_rxnorm_atc.csv` and `eicu_drug_rxnorm_atc.csv`. These are a hard
   gate; harmonization fails without them.
3. Optionally add condition mapping files under `Dataset/mappings/conditions/`
   (`icd10_ccsr.csv`, `icd9_ccs.csv`, `icd9_to_icd10_gem.csv`,
   `icd_chapters.csv`, `eicu_diagnosis_text_condition_map.csv`,
   `project_condition_groups.csv`). These are optional; missing files degrade to
   structural ICD categories and source-native tokens without failing.
   - Run `uv run python scripts/fetch_condition_reference_files.py` (needs
     network) to download authoritative AHRQ CCSR/CCS and CDC GEM sources and
     write `icd10_ccsr.csv`, `icd9_ccs.csv`, `icd9_to_icd10_gem.csv`, and a
     derived `icd_chapters.csv`.
   - Run `uv run python scripts/build_condition_mappings.py` to inventory
     distinct diagnosis concepts and emit review-ready templates for the
     curation-only files (eICU text map, project condition groups).
   - For the approved A1/B3 sepsis deep dive, run
     `uv run python scripts/build_condition_mappings.py --write-curated-sepsis`
     to merge exact ICD codes, `A40`/`A41` ICD prefixes, and discovered eICU
     sepsis text tokens into the active local mapping files before
     harmonization.
4. Run `uv run python -m pipeline.harmonize`.
5. Review `reports/harmonization_manifest.json`,
   `reports/harmonization_coverage.json`, `reports/unmapped_concepts.json`,
   `reports/condition_normalization_coverage.json`, and
   `reports/eicu_diagnosis_text_mapping_review.csv`.
6. Confirm the manifest lists `cohort_stays`, `demographics`, `conditions`,
   `medications`, `labs`, `vitals`, `allergies`, `interventions`, and
   `temporal_events`, and inspect `condition_mapping_resources`.
7. Do not enable pooled training from harmonized artifacts until coverage
   thresholds (see `Documentation/ConditionNormalization.md`) and
   source-specific semantic differences are reviewed.

## Build a Training Table

1. Confirm Milestone 5 harmonization artifacts and aggregate coverage reports
   have been reviewed.
2. Freeze cohort, index time, feature window, label window, and patient split.
3. Run `uv run python -m pipeline.features` to build decision times,
   patient-stay features, and event sequences under
   `Dataset/processed/features/`.
4. Run `uv run python -m pipeline.build_training_table` to build the split
   manifest, train-only candidate catalog, and patient-condition-medication
   table under `Dataset/processed/training/`.
   Use `--candidate-token-strategy atc3_or_rxnorm` or export
   `CANDIDATE_TOKEN_STRATEGY=atc3_or_rxnorm` in the OAR wrappers for a
   coverage-sensitivity rerun that maps candidates to ATC-3 classes before
   falling back to RxNorm ingredients.
5. Build candidates from the training partition only.
6. Create observed-positive and sampled/implicit-negative labels with explicit
   caveats.
7. Exclude future and leakage-prone features by default.
8. Validate one patient belongs to one split.
9. Review `reports/milestone6_feature_manifest.json` and
   `reports/training_table_manifest.json` for censoring, temporal exclusions,
   candidate coverage, out-of-catalog positives, and aggregate-only contents.
   After protected-data materialization, record the gate review in
   `Documentation/Milestone6MaterializationReview.md` and
   `reports/milestone6_materialization_review.json`.

## Run an Experiment

1. Name the hypothesis and baseline.
2. Freeze data and configuration identifiers.
3. Run on validation data while developing.
4. Use test or external-validation data only for the final locked evaluation.
5. Save configuration, metrics, seed, feature list, and model version.
6. Report failures, uncertainty, subgroup limitations, and negative results.
7. Do not promote a poster illustration or exploratory result to a validated
   clinical claim.

## Evaluate Milestone 7 Baselines

1. Confirm Milestone 6 feature and training artifacts exist under
   `$DATASET_ROOT/processed/features/` and `$DATASET_ROOT/processed/training/`.
2. Review `reports/milestone6_feature_manifest.json` and
   `reports/training_table_manifest.json` for censoring, split integrity,
   candidate coverage, out-of-catalog positives, and aggregate-only contents.
3. Run development evaluation with:

```bash
uv run python -m pipeline.evaluate_baselines
```

4. Review `reports/milestone7_coverage_report.json` before interpreting
   performance metrics; eICU splits with zero in-catalog positive groups are
   coverage-only, not external performance.
5. Use validation metrics for baseline comparison and model-selection choices.
6. Run held-out test metrics only after choices are frozen:

```bash
uv run python -m pipeline.evaluate_baselines --mode final --frozen-selection
```

7. Keep row-level scores local under ignored
   `$DATASET_ROOT/processed/evaluation/milestone7/`; report only aggregate
   metrics and coverage.

## Run Phase 8 P0 Feature Ablation

Use this only after the default Milestone 6/7/8B artifacts have been reviewed.
It writes to isolated `phase8_p0` roots and must not overwrite default
canonical roots.

1. Build Phase 8 P0 features:

```bash
uv run python -m pipeline.features --feature-set phase8_p0 \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json" \
  --condition-feature-top-n 40 --trend-min-events 2 \
  --stay-feature-batches 8 --event-sequence-batches 8
```

For protected-data scale:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_features_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_features_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/phase8_p0_features.sh"
```

2. Rebuild the complete CodexPLAN Step 9 model-ready package.

For protected-data scale, submit the dependency-ordered OAR chain through the
submit wrapper:

```bash
scripts/calculco/submit_phase8_p0_model_ready.sh            # full chain
```

The wrapper writes the gitignored `scripts/calculco/phase8_p0_model_ready_job.env`
and the worker sources it. OAR `-S` jobs run in a clean environment and do not
inherit login-shell exports, so setting `PHASE8_P0_START_AT` or the subgraph
knobs with a bare `export` before `oarsub` has no effect on the job. Use the
wrapper (or edit the job env file) instead.

The chain builds or reuses `temporal-features-v2`, then materializes the
RxNorm-first primary training package, the ATC-3-first eICU sensitivity
package, train-only preprocessing, graph edges, normalized patient subgraphs,
local vocabularies, the schema-only data dictionary, and the final aggregate
manifest. Downstream builders infer `feature_version` from their inputs and
fail if version stamps conflict.

Patient nodes are built in eight source-qualified stay-hash batches by default,
keeping every ranking group for one stay together. High-cardinality edge and
candidate joins are independently split into eight shards per node batch. Edge
assembly encodes the small train-fit concept graph and per-shard subgraphs as
integer memberships, builds condition-source and medication-source relations
separately, and uses one DuckDB thread. This avoids the string-key node
self-join that exhausted node-local spill space with 12 million nodes in one
batch. Parts are streamed into the same canonical Parquet schemas.

The aggregate subgraph manifest records both partition levels, part row counts,
and the failed part index. After an edge-stage failure, reuse the completed
training, ATC-3, preprocessing, and graph artifacts by resuming at the subgraph
stage:

```bash
scripts/calculco/submit_phase8_p0_model_ready.sh subgraphs
```

Override the subgraph knobs at submit time, for example:

```bash
SUBGRAPH_JOIN_SHARDS=16 \
  scripts/calculco/submit_phase8_p0_model_ready.sh subgraphs
```

Confirm the new job's `.out` log prints `PHASE8_P0_START_AT=subgraphs` before
trusting the run. The edge builder keeps the large relation-specific joins on
narrow integer keys and attaches the wide string columns only at the end, so the
default 8 shards should fit node-local scratch. `failed to offload data block ...
(X GiB/X GiB used)` means DuckDB hit its `max_temp_directory_size` (which
defaults to ~90% of free space on the temp drive, e.g. a small `/tmp`), not the
`memory_limit`. To resolve it:

- Point `DUCKDB_TEMP_DIR` at a larger volume and raise the ceiling with
  `DUCKDB_MAX_TEMP_DIR_SIZE` (e.g. `150GB`), or
- increase `SUBGRAPH_JOIN_SHARDS` to `16` to shrink each part's working set.

Do not increase `SUBGRAPH_BATCHES` unless node construction itself fails, and
avoid very high shard counts (each part re-scans its node batch, so `32` shards
runs far longer). Prefer a writable `WORK_SCRATCH` through the gitignored local
environment. When unavailable, `common.sh` tries node-local `/scratch` before the
smaller `/tmp` fallback.

The equivalent core login-node commands for synthetic or bounded fixtures are:

```bash
uv run python -m pipeline.build_training_table \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json"

uv run python -m pipeline.build_training_table \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/sensitivity/atc3_or_rxnorm/training" \
  --candidate-token-strategy atc3_or_rxnorm \
  --manifest "$PROJECT_HOME/reports/phase8_p0_atc3_training_table_manifest.json"

uv run python -m pipeline.preprocessing \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --preprocessing-root "$DATASET_ROOT/processed/phase8_p0/training/preprocessing" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_preprocessing_manifest.json"

uv run python -m pipeline.graph_suitability \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --graph-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8" \
  --graph-schema-report "$PROJECT_HOME/reports/phase8_p0_milestone8_graph_schema.json" \
  --suitability-report "$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json" \
  --ablation-plan "$PROJECT_HOME/reports/phase8_p0_milestone8_ablation_plan.json"

uv run python -m pipeline.patient_subgraphs \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --graph-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8" \
  --subgraphs-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8/patient_subgraphs" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_patient_subgraphs_manifest.json" \
  --subgraph-batches 8 \
  --subgraph-join-shards 8 \
  --edge-duckdb-threads 1

uv run python -m pipeline.model_ready_package \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --graph-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8" \
  --subgraphs-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8/patient_subgraphs" \
  --preprocessing-root "$DATASET_ROOT/processed/phase8_p0/training/preprocessing" \
  --package-root "$DATASET_ROOT/processed/phase8_p0/model_ready" \
  --data-dictionary "$PROJECT_HOME/reports/phase8_p0_model_ready_data_dictionary.json" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_model_ready_manifest.json" \
  --primary-training-manifest "$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json" \
  --sensitivity-training-manifest "$PROJECT_HOME/reports/phase8_p0_atc3_training_table_manifest.json" \
  --preprocessing-manifest "$PROJECT_HOME/reports/phase8_p0_preprocessing_manifest.json" \
  --subgraphs-manifest "$PROJECT_HOME/reports/phase8_p0_patient_subgraphs_manifest.json"
```

Do not run `pipeline.model_ready_package` before graph suitability and the
ATC-3 sensitivity build; both are required by the completion manifest. If both
candidate strategies have zero positive eICU ranking groups, the final
manifest records `coverage_only_no_in_catalog_positive_groups` and prohibits
external performance claims.

3. Rerun Milestone 7 development evaluation on the isolated roots:

```bash
uv run python -m pipeline.evaluate_baselines \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --evaluation-root "$DATASET_ROOT/processed/phase8_p0/evaluation/milestone7" \
  --coverage-report "$PROJECT_HOME/reports/phase8_p0_milestone7_coverage_report.json" \
  --evaluation-report "$PROJECT_HOME/reports/phase8_p0_milestone7_baseline_evaluation.json" \
  --training-manifest "$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json"
```

For protected-data scale, submit through OAR:

```bash
scripts/calculco/submit_phase8_p0_evaluate_baselines.sh development
```

`submit_phase8_p0_evaluate_baselines.sh` writes
`scripts/calculco/phase8_p0_milestone7_job.env` (gitignored) before `oarsub`.
Override defaults at submit time, for example:

```bash
MILESTONE7_BASELINES=linear,xgboost \
  scripts/calculco/submit_phase8_p0_evaluate_baselines.sh development
```

4. Rerun Milestone 8B development gates. The model-ready chain already builds
   the isolated Milestone 8 graph; rerun graph suitability only when its inputs
   changed:

```bash
uv run python -m pipeline.graph_suitability \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --graph-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8" \
  --graph-schema-report "$PROJECT_HOME/reports/phase8_p0_milestone8_graph_schema.json" \
  --suitability-report "$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json" \
  --ablation-plan "$PROJECT_HOME/reports/phase8_p0_milestone8_ablation_plan.json"

uv run python -m pipeline.graph_ablation --mode development \
  --allow-development-milestone7-reference \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --training-root "$DATASET_ROOT/processed/phase8_p0/training" \
  --graph-root "$DATASET_ROOT/processed/phase8_p0/graph/milestone8" \
  --milestone7-evaluation-root "$DATASET_ROOT/processed/phase8_p0/evaluation/milestone7" \
  --evaluation-root "$DATASET_ROOT/processed/phase8_p0/evaluation/milestone8b" \
  --evaluation-report "$PROJECT_HOME/reports/phase8_p0_milestone8b_ablation_evaluation.json" \
  --frozen-selection-report "$PROJECT_HOME/reports/phase8_p0_milestone8b_frozen_selection.json" \
  --feature-manifest "$PROJECT_HOME/reports/phase8_p0_milestone8b_graph_feature_manifest.json" \
  --milestone6-feature-manifest "$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json" \
  --training-manifest "$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json" \
  --milestone7-evaluation-report "$PROJECT_HOME/reports/phase8_p0_milestone7_baseline_evaluation.json" \
  --milestone8-suitability-report "$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json"
```

For protected-data scale, submit through OAR. Recommended one-shot chain after job 6612:

```bash
cd "$PROJECT_HOME"
scripts/calculco/submit_phase8_p0_graph_gates.sh
```

Or run the steps separately:

```bash
scripts/calculco/submit_phase8_p0_graph_suitability.sh
scripts/calculco/submit_phase8_p0_graph_ablation.sh development
```

5. Write the aggregate promotion gate review:

```bash
scripts/calculco/phase8_p0_feature_gate_review.sh
```

Equivalent login-node command:

```bash
uv run python -m pipeline.feature_gate_review \
  --phase8-evaluation-report "$PROJECT_HOME/reports/phase8_p0_milestone8b_ablation_evaluation.json" \
  --output "$PROJECT_HOME/reports/phase8_p0_feature_gate_review.json"
```

Promote only after reviewing the gate JSON. If it rejects or is inconclusive,
keep the default roots and current `milestone8b_*` reports canonical.

## Run Phase 8 P0 Gate Recovery

1. Confirm the completed model-ready and frozen reference reports exist. Do
   not inspect patient rows.
2. Submit the CPU-only development job:

```bash
cd "$PROJECT_HOME"
scripts/calculco/submit_phase8_p0_gate_recovery.sh development
```

The worker runs `pipeline.training_contract` first. The first run creates
`reports/phase8_p0_training_contract_lock.json`; later runs compare current
manifest hashes and artifact metadata with that lock and fail on drift. The
rank-aware runner then makes every feature, hyperparameter, and fusion choice
on deterministic patient-grouped MIMIC-train folds.

Development mode is memory-bounded: configuration screening runs on the
deterministic train sample, while the locked finalist, binary reference OOF, and
the validation gate use the full candidate universe streamed to narrow Parquet
one fold at a time, with the fusion-weight search executed in DuckDB. The worker
logs stage progress and peak RSS to the OAR `.out`, writes a contract-keyed
`screening_checkpoint.json` so a resubmit resumes after screening, and defaults
to `DUCKDB_THREADS=8`, `DUCKDB_MEMORY_LIMIT=24GB`, `MALLOC_ARENA_MAX=2`, and
`PYTHONUNBUFFERED=1`. The DuckDB ceiling stays below the node budget so pandas,
sklearn, and XGBoost keep headroom; ensure DuckDB spill lands on disk-backed
scratch rather than a `/tmp` tmpfs (the worker warns on `/tmp` fallback).

3. Review only these aggregate reports:

```text
reports/phase8_p0_training_contract_lock.json
reports/phase8_p0_training_contract_audit_latest.json
reports/phase8_p0_gate_recovery_evaluation.json
reports/phase8_p0_gate_recovery_selection.json
```

Local sampled matrices, full scores, preprocessors, and models remain ignored
under `$DATASET_ROOT/processed/phase8_p0/evaluation/gate_recovery/`.

4. Run final mode only if the selection report records
`neural_training_authorized=true`:

```bash
scripts/calculco/submit_phase8_p0_gate_recovery.sh final
```

The CLI independently blocks final MIMIC test scoring when the development
gate failed. eICU is not used for fitting or tuning.

## Run Phase 8 P0 Neural Training (Stage 2)

Run the Stage 2 Transformer patient/context branch after the structured
recovery gate records `neural_training_authorized=true` in
`reports/phase8_p0_gate_recovery_selection.json` (Stage 1 development has
already frozen that authorization). This section trains the Transformer branch;
the separate implemented Phase D workflow is documented below.

1. Confirm the gate selection is frozen/authorized and the model-ready
   artifacts, contract lock, and Stage 1 recovery baseline scores exist under
   `$DATASET_ROOT/processed/phase8_p0/evaluation/gate_recovery/baseline_scores.parquet`.
   Do not inspect patient rows. The neural gate beats
   `xgboost_rank_ndcg_oof_late_fusion` (≈ `0.394607` NDCG@10), not the older
   Milestone 8B `xgboost_frozen_reference`.
2. Install the optional PyTorch group on the worker (it is not synced by
   default so the rest of the repository stays lightweight):

```bash
uv sync --group neural
```

3. Prepare train-fit vocabularies, numeric normalization, and memory-bounded
   sharded DuckDB caches, then train and score:

```bash
uv run python -m pipeline.neural_training prepare
uv run python -m pipeline.neural_training train
uv run python -m pipeline.neural_training score
```

The `prepare` stage is PyTorch-free and can run on the login node for bounded
fixtures; `train` and `score` import torch lazily and expect a GPU/CPU worker.
Every stage runs the same fail-closed preflight (`contract` module): it verifies
the training-contract lock and re-checks the neural gate. `prepare` builds
vocabularies with reserved `PAD=0`/`UNK=1` tokens, computes train-only numeric
and event-value normalization, fits train-only candidate priors
(global and condition×candidate log-odds) plus `log1p(candidate_rank)` and
Stage-1-matched train-fit graph tabular side features (support threshold 5),
and writes `feature_layout.json` so training and scoring share one feature
ordering. After any schema/prior/graph-side change (including the v3
gap-recovery upgrade), **rerun `prepare` before `train`** so group caches
contain the new candidate-side columns. `train` runs multi-positive listwise
softmax, catalog-primary-positive listwise (MRR-oriented), and weighted
auxiliary BCE (per-batch `pos_weight`) with AdamW warmup+cosine, EMA weight
selection, early stopping on validation NDCG@10, gradient clipping, optional
mixed precision, checkpointing, and post-hoc temperature calibration. `score`
writes canonical `baseline_scores.parquet` rows, combines them with the Stage 1
recovery baseline scores for authoritative DuckDB metrics, and records the
neural gate decision against that winner (+0.005 NDCG@10).

4. For protected-data scale, submit the GPU OAR wrapper (it installs the
   `neural` group and runs the selected stages):

```bash
scripts/calculco/submit_phase8_p0_neural_training.sh development
```

The worker requests `/nodes=1/gpu=1` plus `#OAR -p gpudevice<>'-1'` (no extra
quotes around the `-p` expression) so OAR places the job on a GPU host
(chimay31+ / visu*), not a CPU node such as chimay01. Confirm
`assigned_hostnames` with `oarstat -j <id> -f` / `oarstatmon.py` before trusting
a long train run. To pin one host, set
`#OAR -p network_address='chimay34' and gpudevice<>'-1'` in
`phase8_p0_neural_training.sh`. Tune stages, mode, seed, sequence length, batch
size, and DuckDB limits through the generated
`scripts/calculco/phase8_p0_neural_training_job.env`.

5. Review only these aggregate reports:

```text
reports/phase8_p0_neural_prepare_manifest.json
reports/phase8_p0_neural_training_evaluation.json
reports/phase8_p0_neural_score_evaluation.json
reports/phase8_p0_neural_training_selection.json
```

Local caches, checkpoints, predictions, and row-level scores remain ignored
under `$DATASET_ROOT/processed/phase8_p0/neural/`.

6. Run final MIMIC test scoring only after the development selection records a
   passing neural gate and the neural selection is frozen (`--mode final
   --frozen-selection`). eICU is not used for fitting, tuning, or selection.

## Run Phase 8 P0 GNN and Frozen-Transformer Fusion (Phase D)

Phase D code and protected preparation are complete, but protected training has
not completed successfully. The full-train graph is refit-only; model
selection requires five patient-fold-excluded graph caches and exact artifact
hashes.

1. Export the Calculco paths and sync the existing optional PyTorch group:

```bash
export PROJECT_HOME=/path/to/ResearchModule
export DATASET_ROOT=/path/to/protected/Dataset
export WORK_SCRATCH=/path/to/job/scratch
export REPORTS_ROOT="${REPORTS_ROOT:-$PROJECT_HOME/reports}"
cd "$PROJECT_HOME"
uv sync --group neural
```

2. Confirm the upstream aggregate contracts and frozen Transformer artifacts
   exist without opening patient rows:

```bash
test -f "$REPORTS_ROOT/phase8_p0_training_contract_lock.json"
test -f "$REPORTS_ROOT/phase8_p0_patient_subgraphs_manifest.json"
test -f "$REPORTS_ROOT/phase8_p0_neural_training_selection.json"
test -f "$DATASET_ROOT/processed/phase8_p0/neural/checkpoints/transformer_recommender.pt"
uv run python -m pipeline.gnn_training --help
```

3. Review protected-NFS free space. Cross-fit preparation currently writes five
   physical fold trees, so set `GNN_CROSSFIT_MIN_FREE_GIB` to the
   reviewed minimum; Python also computes a cache-derived estimate and requires
   whichever value is larger. Submit the CPU preparation job:

```bash
export WORK_SCRATCH=/workdir/<lab>/<username>
# Defaults after the DuckDB OOM on job 7800: 4 threads, 128GB DuckDB ceiling,
# and a large spill cap on WORK_SCRATCH. Override only after capacity review.
GNN_CROSSFIT_MIN_FREE_GIB=<reviewed-value> \
  scripts/calculco/submit_phase8_p0_gnn_prepare.sh
```

The wrapper requires DuckDB spill under `$WORK_SCRATCH`, runs
`uv sync --group neural`, and executes only `prepare`. Cross-fit rebuilds the
~9.5GB patient-subgraph edge table one ranking-group shard at a time so joins
can spill instead of materializing the full edge set in RAM. It
does not print patient rows. Preparation is ready only when both reports record
`status=completed`:

```text
reports/phase8_p0_gnn_prepare_manifest.json
reports/phase8_p0_gnn_crossfit_graph_manifest.json
```

Restricted graph caches, exact hash manifests, frozen Transformer contexts and
logits, and vocabularies stay under
`$DATASET_ROOT/processed/phase8_p0/gnn/`.

4. The P1 feature/relation versions invalidate the earlier prepared GNN tree.
   Rerun and review `prepare` before GPU work. Then submit the restartable
   `(variant, fold)` OAR array:

```bash
scripts/calculco/submit_phase8_p0_gnn_fold_array.sh
```

After every array task has a compatible completed-fold checkpoint and hash
sidecar manifest, run
selection and full-train refit as separate jobs, then submit later stages
individually:

```bash
scripts/calculco/submit_phase8_p0_gnn_training.sh development select-gnn
scripts/calculco/submit_phase8_p0_gnn_training.sh development refit-gnn
scripts/calculco/submit_phase8_p0_gnn_training.sh development score-gnn
scripts/calculco/submit_phase8_p0_gnn_training.sh development train-fusion
scripts/calculco/submit_phase8_p0_gnn_training.sh development score-fusion
```

Set `GNN_AFTER_JOB_ID=<prior OAR job id>` on each submission to add an OAR
anterior-job dependency (`oarsub -a`), including selection after the fold
array, refit after selection, and each reviewed scoring/fusion boundary.
Note that `-a` starts the next job when the prior **ends**, including on
failure; prefer success-gated watchers when a failed prior must block the
next stage.

For end-of-job alerts, set `GNN_OAR_NOTIFY` before submit (passed to
`oarsub --notify`), for example
`GNN_OAR_NOTIFY='[END,ERROR]mail:you@example.com'`. For a job already
running, use `scripts/calculco/watch_oar_job.sh <job_id> --mail you@example.com`
(optional `--on-success '…'` to submit the next stage only when
`exit_code` is 0).

The compatibility `train-gnn` command still performs missing folds, selection,
and refit serially, but the OAR array is the protected-scale workflow. Six
pre-registered variants include full, rank-only, no message passing, and three
relation-removal controls. `select-gnn` writes selected GNN OOF logits and fits
temperature from those patient-grouped train OOF logits; `refit-gnn` trains the
selected variant on the full train graph. `score-gnn`
compares the standalone branch with the locked graph-only XGBoost reference.
Fusion keeps the Transformer immutable and compares late versus residual
candidates against the frozen Transformer validation metrics. Late-fusion
alpha is fit on the intersection of patient-fold GNN OOF logits with the
full-train frozen Transformer cache; that cache excludes zero-positive train
groups by design, matching the refit graph.

Calculco A100 jobs default to BF16 (`GNN_PRECISION=bf16`); FP16 remains an
explicit option and alone uses dynamic loss scaling and bounded overflow
retry. Use `GNN_PRECISION=fp32` or the legacy `GNN_MIXED_PRECISION=0` for a
separately reviewed full-precision diagnostic. Batches are greedily bounded by
`GNN_BATCH_RANKING_GROUPS=32`, `GNN_MAX_EDGES_PER_BATCH=100000`, and
`GNN_MAX_NODES_PER_BATCH=8192`. Microbatches accumulate to
`GNN_GRADIENT_ACCUMULATION_GROUPS=32` before each optimizer step; tune ceilings
from measured aggregate distributions, not patient examples. New prepared
caches enforce one Parquet fragment per table/split/shard. The complete tree is
content-hashed once per allocation; later preflights in that allocation reuse
a mode-0600 attestation after a metadata fingerprint check. Training reports
one aggregate-safe heartbeat every
`GNN_PROGRESS_INTERVAL_BATCHES=100` batches.

The relation layer aggregates edges by relation before applying each dense
transform and therefore does not materialize an `E×H×H` tensor. Atomic epoch
resume state and completed-fold checkpoints live under the configured GNN
checkpoint root, outside the immutable cross-fit cache. Resume is accepted
only for matching seed, layout, architecture, optimization, and cross-fit
manifest locks.

The frozen Transformer representations are a full-train refit, not
Transformer OOF outputs. Late-fusion weight fitting is recorded as train
meta-fitting over GNN OOF logits and a fixed train-derived Transformer
covariate; only the separate validation gate supports promotion.

5. Review aggregate reports only:

```text
reports/phase8_p0_gnn_training_evaluation.json
reports/phase8_p0_gnn_score_evaluation.json
reports/phase8_p0_gnn_training_selection.json
reports/phase8_p0_fusion_training_evaluation.json
reports/phase8_p0_fusion_score_evaluation.json
reports/phase8_p0_fusion_training_selection.json
```

6. Final scoring is optional, gated, atomic, and one-shot. Submit exactly one
   selected scorer only after its development selection authorizes final
   scoring:

```bash
GNN_FINAL_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT \
  scripts/calculco/submit_phase8_p0_gnn_training.sh final score-gnn

# Or, for an authorized promoted hybrid:
GNN_FINAL_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT \
  scripts/calculco/submit_phase8_p0_gnn_training.sh final score-fusion
```

The scoring command creates an exclusive completion/attempt marker before
test predictions. A concurrent run or interrupted attempt remains blocked; do
not remove or reset the marker without a separate reviewed recovery decision.
The broader MIMIC test split is not wholly unseen because the frozen
Transformer was evaluated earlier, and reports disclose that limitation.

## Run Paired-OOF Late Fusion (Protocol v2)

Use this workflow to replace the asymmetric original late-fusion meta-fit. The
five Transformer folds and six GNN-variant OOF materializations are independent
and are submitted as two parallel OAR arrays:

```bash
export PROJECT_HOME DATASET_ROOT WORK_SCRATCH
scripts/calculco/submit_phase8_p0_paired_oof.sh all-oof
```

Each Transformer task rebuilds fit-only preprocessing and uses the matching
fold-excluded graph. Each GNN task scores all five held-out checkpoints for one
variant. Do not submit selection until all eleven aggregate reports say
`status=completed`.

For a failed individual task, use the targeted retry boundary rather than
repeating an array:

```bash
scripts/calculco/submit_phase8_p0_paired_oof.sh transformer-fold 1
scripts/calculco/submit_phase8_p0_paired_oof.sh gnn-variant full
```

The GPU worker reserves 48 hours, is restricted to the proven
`chimay33`/`chimay34` pool, and fails before pipeline work when CUDA is not
visible. The first full GNN materializer exceeded the former 12-hour limit, so
do not use the original 6--10 hour planning estimate. Submit additional GNN
variants only after reviewing the first recovery task's completion and runtime.

To run the remaining variants serially with mail alerts and automatic hand-off,
start a login-node watcher inside `tmux` or `screen`:

```bash
export PROJECT_HOME DATASET_ROOT WORK_SCRATCH
PAIRED_GNN_CHAIN_MAIL=you@example.com \
  scripts/calculco/watch_paired_gnn_oof_chain.sh
```

The chain skips variants whose aggregate report already says `status=completed`,
submits the next pending variant only after the prior job finishes with exit
code 0 **and** the aggregate report says `completed`, and mails on every job end.
On failure it stops instead of continuing. Optional auto-selection after the
last variant:

```bash
PAIRED_GNN_CHAIN_MAIL=you@example.com \
PAIRED_GNN_CHAIN_ON_COMPLETE='scripts/calculco/submit_phase8_p0_paired_oof.sh select' \
  scripts/calculco/watch_paired_gnn_oof_chain.sh
```

For one variant at a time without the chain, use `gnn-variant VARIANT`. For OAR
dependencies on a future submit, set `PAIRED_AFTER_JOB_ID=<prior job id>`
(starts even on failure, like `GNN_AFTER_JOB_ID`). For submit-time OAR notify,
set `PAIRED_OAR_NOTIFY='[END,ERROR]mail:you@example.com'`. If built-in OAR
notify is unreliable, prefer the watcher; it uses `send_oar_mail.sh` with a
filesystem fallback when `mail`/`mailx` fail. Test mail once:

```bash
scripts/calculco/send_oar_mail.sh you@example.com "Calculco OAR test" "Watcher mail works."
```

```bash
scripts/calculco/submit_phase8_p0_paired_oof.sh select
scripts/calculco/submit_phase8_p0_paired_oof.sh refit
```

`select` is a CPU OAR job. It requires exact fold/candidate/label alignment and
searches the locked alpha grid `0.000..0.250` by `0.005` on paired train OOF
only. `refit` reuses the existing full GNN when variant and epoch count match;
otherwise it runs one selected-variant refit.

The final boundary is blocked until a reviewed newly isolated gate and matching
cache partitions exist. The pipeline never chooses that split automatically.
After freezing `reports/phase8_p0_paired_oof_frozen_gate.json`:

```bash
PAIRED_GATE_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT \
  scripts/calculco/submit_phase8_p0_paired_oof.sh score
```

The command claims the gate before scoring and applies the unchanged `+0.005`
NDCG@10 / `-0.01` secondary guards. See
`Documentation/PairedOOFLateFusionProtocol.md` for the manifest contract,
artifact layout, measured prior runtimes, and planning ranges.

## Run Milestone 8 Graph Suitability

1. Confirm Milestone 6 feature/training artifacts exist and Milestone 7 frozen
   selection is recorded.
2. Run graph-readiness analysis with:

```bash
uv run python -m pipeline.graph_suitability
```

3. For protected-data scale, submit through OAR:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/graph_suitability.sh"
```

4. Review `reports/milestone8_graph_schema.json`,
   `reports/milestone8_graph_suitability.json`, and
   `reports/milestone8_ablation_plan.json`.
5. Keep concept-level graph edges local under ignored
   `$DATASET_ROOT/processed/graph/milestone8/`.
6. Treat this as an upstream graph gate only. Phase D still requires its own
   protected preparation, validation selection, and frozen-artifact review
   before any Transformer-GNN improvement claim.

## Run Milestone 8B Graph Ablations

1. Confirm Milestone 7 final evaluation and Milestone 8 graph suitability have
   completed:

```bash
test -f "$PROJECT_HOME/reports/milestone7_baseline_evaluation.json"
test -f "$PROJECT_HOME/reports/milestone8_graph_suitability.json"
test -f "$DATASET_ROOT/processed/graph/milestone8/graph_edges.parquet"
```

2. Run development ablation locally for small fixtures or through OAR for
   protected-data scale:

```bash
uv run python -m pipeline.graph_ablation
scripts/calculco/submit_graph_ablation.sh development
```

3. Review aggregate-only reports:
   `reports/milestone8b_graph_feature_manifest.json`,
   `reports/milestone8b_ablation_evaluation.json`, and
   `reports/milestone8b_frozen_selection.json`.
4. Run final MIMIC test metrics only after the frozen 8B selection exists:

```bash
scripts/calculco/submit_graph_ablation.sh final
```

5. Keep local row-level features, scores, models, and fusion weights under
   ignored `$DATASET_ROOT/processed/evaluation/milestone8b/`.
6. Do not interpret Milestone 8B as a clinical recommendation result, full
   Transformer-GNN model, or valid eICU external performance result while eICU
   has zero in-catalog positives.

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

Use the cluster for heavy extraction, modeling, and long-running jobs.

1. Confirm `PROJECT_HOME` and `DATASET_ROOT` are exported.
2. Read account-specific paths in gitignored `Documentation/CalculcoSetup.local.md`
   (create from `Documentation/CalculcoSetup.example.md`). Generic platform notes:
   `Documentation/CalculcoSetup.md`.
3. Export runtime paths via `.env.calculco` or `scripts/calculco/common.local.sh`
   (both gitignored) before pipeline work.
4. Use `$WORK_SCRATCH/runs` for temporary job I/O; copy results back to protected
   storage after jobs complete.
5. Transfer large files with `rsync` via `pcsdata.univ-littoral.fr`.
6. Submit compute with OAR (`oarsub`), not on the login node.
7. Load software with `module load` or `uv` as documented on the Calculco website.

### Run Source Extraction on Calculco

Repository OAR scripts live under `scripts/calculco/`. `common.sh` loads
`.env.calculco` / `common.local.sh`, requires `DATASET_ROOT`, and sets scratch
for `TMPDIR` and `UV_CACHE_DIR` when `WORK_SCRATCH` is set. Outputs go to
`$DATASET_ROOT/processed/extracts/`; manifests to `$PROJECT_HOME/reports/`.

Preflight on the login node:

```bash
test -f "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
test -f "$PROJECT_HOME/reports/quality_profile.json"
test -f "$PROJECT_HOME/reports/source_integrity_failed_tables.json"
cd "$PROJECT_HOME" && uv run python -V
```

Submit (pass log paths at submit time — not hard-coded in Git):

```bash
chmod +x scripts/calculco/*.sh
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_extract_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_extract_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic_eicu.sh"

# or run MIMIC and eICU in parallel:
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_eicu_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_eicu_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_eicu.sh"
```

Monitor with `oarstatmon.py` (overview) or `oarstat`. During the 2026 platform
migration, see `Documentation/CalculcoSetup.md` for `ritchie` login and node
property changes. Review only aggregate manifests after completion:

### Run Harmonization on Calculco

Outputs go to `$DATASET_ROOT/processed/harmonized/`; reports to
`$PROJECT_HOME/reports/`. Medication mapping files are a **hard gate**;
condition roll-up files under `$DATASET_ROOT/mappings/conditions/` are optional
(fetch with `uv run python scripts/fetch_condition_reference_files.py` on the
login node before submit).

Preflight on the login node:

```bash
test -f "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
test -f "$DATASET_ROOT/processed/extracts/mimiciv/diagnoses_icd.parquet"
test -f "$DATASET_ROOT/processed/extracts/eicu_crd/diagnosis.parquet"
test -f "$DATASET_ROOT/mappings/medications/mimic_ndc_rxnorm_atc.csv"
test -f "$DATASET_ROOT/mappings/medications/eicu_drug_rxnorm_atc.csv"
cd "$PROJECT_HOME" && uv run python -V
```

Submit:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/harmonize.sh"
```

`harmonize.sh` defaults to conservative memory settings for the migration-era
OAR cgroup (`DUCKDB_THREADS=4`, `DUCKDB_MEMORY_LIMIT=10GB`) and splits the large
lab/vital materialization path with `HARMONIZE_DOMAIN_BATCHES=4`. If labs or
vitals still fail with DuckDB OOM, resubmit with a higher batch count before
requesting a bigger node:

```bash
export HARMONIZE_DOMAIN_BATCHES=8
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/harmonize.sh"
```

The manifest records `build_strategy`, `batch_count`, part counts, and whether a
stale final output existed when a split materialization failed. Treat any
non-`completed` manifest as unusable for Milestone 6, even if older Parquet
files are still present on disk.

If submission fails with **"There are not enough resources"**, run
`oarstatmon.py`. On the legacy `calculco` front-end during the 2026 migration,
most CPU nodes may be **Dead** while `gpudevice='-1'` excludes the only **Alive**
GPU nodes — see [OAR troubleshooting](#oar-troubleshooting-calculco-migration)
below. `harmonize.sh` requests `gpudevice='-1'` for CPU-only placement.

Review only aggregate manifests after completion:

- `reports/harmonization_manifest.json`
- `reports/harmonization_coverage.json`
- `reports/condition_normalization_coverage.json`
- `reports/unmapped_concepts.json`

### Re-profile source tables on Calculco

Re-run the **full** aggregate quality profile after correcting local
`chartevents` / `inputevents` source files so their `scan_failed` entries
refresh and the extraction gates can materialize `inputevents`. A full run is
required because `pipeline.profile_tables` rewrites the whole
`reports/quality_profile.json`; a `--table` subset would drop the other tables'
gate entries.

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/profile_tables.sh"
```

Confirm `mimic_chartevents` and `mimic_inputevents` are `completed` in
`reports/quality_profile.json`, then re-run the MIMIC extractor so
`inputevents` materializes past its gate.

### Run Milestone 6 feature and training builds on Calculco

Run only after Milestone 5 harmonization and its aggregate coverage reports are
reviewed. Outputs go to `$DATASET_ROOT/processed/features/` and
`$DATASET_ROOT/processed/training/`; manifests to `$PROJECT_HOME/reports/`.

Preflight on the login node:

```bash
for t in cohort_stays demographics conditions medications labs vitals \
  allergies interventions temporal_events; do
  test -f "$DATASET_ROOT/processed/harmonized/$t.parquet" || echo "MISSING $t"
done
```

Submit the full Milestone 6 chain (features then training table) in one job:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/milestone6.sh"
```

`patient_stay_features` and `event_sequences` are materialized in stay-hash
batches. The OAR wrappers default to `DUCKDB_THREADS=4`,
`DUCKDB_MEMORY_LIMIT=10GB`, `STAY_FEATURE_BATCHES=8`, and
`EVENT_SEQUENCE_BATCHES=8`. Export `STAY_FEATURE_BATCHES=16` if lab/vital
feature aggregation still runs out of memory; export `EVENT_SEQUENCE_BATCHES=16`
if event-sequence windowing still runs out of memory.

For an out-of-catalog-positive coverage sensitivity run, export
`CANDIDATE_TOKEN_STRATEGY=atc3_or_rxnorm` before `milestone6.sh` or
`build_training_table.sh`. This keeps candidate fitting train-only while using
ATC-3 class tokens when available.

Or run the stages as separate jobs (`build_training_table.sh` depends on the
feature artifacts from `features.sh`):

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/features.sh"
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/build_training_table.sh"
```

Review only aggregate manifests after completion:

- `reports/milestone6_feature_manifest.json` (eligibility, splits, temporal
  exclusions)
- `reports/training_table_manifest.json` (split integrity, candidate counts,
  training rows by split, out-of-catalog positives, coverage losses)

Run Milestone 7 coverage and baseline evaluation after Milestone 6 review:

```bash
scripts/calculco/submit_evaluate_baselines.sh development
```

`submit_evaluate_baselines.sh` writes `scripts/calculco/milestone7_job.env`
(gitignored) before `oarsub` so the worker receives the intended baseline list
and mode. Override defaults at submit time, for example:

```bash
MILESTONE7_BASELINES=random,global_popularity,condition_popularity,linear \
  scripts/calculco/submit_evaluate_baselines.sh development
```

Optional environment controls (also settable in `milestone7_job.env.example`):

```bash
export MILESTONE7_TOP_K=1,3,5,10
export MILESTONE7_CONDITION_TOKENS=condition:sepsis
```

After reviewing `reports/milestone7_validation_summary.json` and confirming
`reports/milestone7_frozen_selection.json`, submit held-out test evaluation:

```bash
scripts/calculco/submit_evaluate_baselines.sh final
```

### OAR troubleshooting (Calculco migration)

```bash
oarstatmon.py
```

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `not enough resources` + type constraints | CPU nodes Dead; `gpudevice='-1'` leaves no nodes | SSH to `ritchie.univ-littoral.fr`, or omit `gpudevice='-1'` in the script |
| Job waits (`W`) a long time | Cluster busy | Try `-t besteffort` or fewer cores |

Probe that scheduling works (then `oardel <jobid>`):

```bash
oarsub -l /nodes=1/core=4,walltime=1:00:00 -t besteffort \
       -O /tmp/oar_probe_%jobid%.out -E /tmp/oar_probe_%jobid%.err "echo ok"
```

Job stdout/stderr are written under `scripts/calculco/logs/` and are gitignored.
Do not paste patient-level extract rows into chat, docs, or version control.
