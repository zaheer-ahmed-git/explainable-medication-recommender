# Milestone 7 Execution Plan: Baselines and Evaluation

## Summary

Milestone 7 should implement transparent medication-ranking baselines over the completed Milestone 6 artifacts, then produce aggregate-only evaluation reports. Priority is: coverage sanity first, non-learned baselines second, learned baselines third, final held-out evaluation last.

Current gating facts to respect:

- Milestone 6 artifacts are complete: feature, split, candidate catalog, and `patient_condition_medication` tables exist.
- MIMIC train/validation/test splits are patient-level; eICU is external.
- eICU currently has `0` in-catalog positive rows, so external performance metrics are not valid yet; report eICU as coverage/compatibility analysis until mappings/catalog overlap improve.
- Sepsis B3 is approved for Milestone 7 headline claims. The local mapping
  writer exists (`scripts/build_condition_mappings.py --write-curated-sepsis`),
  but harmonization and Milestone 6 artifacts must be refreshed before B3
  headline evaluation.

## Current Implementation Status

P0-P4 implementation is present in `pipeline.evaluate_baselines` and
`pipeline.learned_baselines`.
Implemented now:

- aggregate `milestone7_coverage_report.json` coverage/evaluability reporting;
- aggregate `milestone7_baseline_evaluation.json` reporting;
- local ignored row-level scores and learned model artifacts under
  `$DATASET_ROOT/processed/evaluation/milestone7/`;
- deterministic random, global-popularity, and condition-popularity baselines;
- learned linear (`SGDClassifier`) and XGBoost baselines with train-only
  positive + deterministic 5:1 weak-negative hash-threshold sampling
  stratified by condition before joining wide stay features;
- AP, ROC-AUC, Brier, 10-bin ECE, precision@k, recall@k, hit_rate@k,
  NDCG@k, and MRR@k;
- suppression of per-condition metrics below the positive-group threshold;
- final/test metric blocking unless `--mode final --frozen-selection` is
  explicit, with final-mode protected-data metrics recorded after frozen
  selection;
- Calculco wrapper `scripts/calculco/evaluate_baselines.sh`;
- synthetic tests in `tests/test_milestone7_baselines.py`.

Still pending:

- sepsis/B3 headline report after running the curated sepsis mapping writer and
  refreshing harmonized/Milestone 6 artifacts;
- eICU external performance metrics if in-catalog positives become evaluable.

## Protected-Data Development Results (Job 2084)

Development-mode protected-data evaluation completed on Calculco job **2084**
(~28 min, `status=completed`). Aggregate references:

- `reports/milestone7_validation_summary.json`
- `reports/milestone7_frozen_selection.json`
- `reports/milestone7_baseline_evaluation.json` (development mode)

MIMIC validation ranking @10 (57,810 positive groups):

| Baseline | Hit@10 | NDCG@10 | MRR@10 | ROC-AUC |
|----------|--------|---------|--------|---------|
| xgboost | 0.848 | 0.370 | 0.488 | 0.705 |
| condition_popularity | 0.804 | 0.299 | 0.398 | 0.639 |
| linear | 0.808 | 0.283 | 0.369 | 0.620 |
| global_popularity | 0.803 | 0.274 | 0.360 | 0.616 |
| random | 0.559 | 0.141 | 0.201 | 0.500 |

Coverage caveats remain in force: ~50% MIMIC positives are out-of-catalog; eICU
has zero in-catalog positive groups (coverage-only, not external performance).

## Frozen Selection (P4 Gate)

Validation winner for headline comparison: **`xgboost`** (fixed v1
hyperparameters, no validation tuning loop). Recorded in
`reports/milestone7_frozen_selection.json` on 2026-07-07.

Held-out MIMIC test metrics were produced after this frozen-selection record
existed. The current final-mode aggregate report is
`reports/milestone7_baseline_evaluation.json`.

Future reruns must still use:

```bash
scripts/calculco/submit_evaluate_baselines.sh final
```

Do not interpret development validation metrics as clinical performance.

## Implementation Changes

- Add a Milestone 7 evaluation CLI, preferably `pipeline.evaluate_baselines`, with defaults:
  - inputs: `$DATASET_ROOT/processed/features/` and `$DATASET_ROOT/processed/training/`;
  - outputs: local row-level scores/models under `$DATASET_ROOT/processed/evaluation/milestone7/`;
  - aggregate reports under `$PROJECT_HOME/reports/`;
  - top-k defaults: `1,3,5,10`;
  - development mode evaluates train diagnostics + validation only;
  - final mode requires an explicit frozen-selection flag before test metrics are produced.
- Add config constants for evaluation output/versioning, for example `EVALUATION_ROOT`, `BASELINE_VERSION`, and `EVALUATION_VERSION`.
- Implement train-only baselines:
  - deterministic random score from stable hash of seed + ranking group + candidate;
  - global medication popularity from MIMIC train labels only;
  - condition-specific popularity from MIMIC train labels only;
  - linear baseline using `SGDClassifier(loss="log_loss")` on sampled MIMIC train rows;
  - XGBoost baseline using existing `xgboost` dependency with fixed v1 hyperparameters.
- Fit learned baselines on all train positives plus a deterministic 5:1 weak-negative sample, stratified by condition; score all validation/test rows without downsampling.
- Use only pre-decision features plus candidate/condition metadata:
  - allowed: demographics, admission context, lab/vital summaries, allergy/intervention counts, `index_condition_token`, `candidate_medication_token`, `candidate_rank`;
  - excluded: patient/stay identifiers as predictors, label columns, post-decision timestamps, label event counts, outcome fields, medication-history features.
- Add an OAR wrapper `scripts/calculco/evaluate_baselines.sh` using existing `common.sh`, with preflight checks for Milestone 6 artifacts and bounded DuckDB scratch/memory settings.

## Phases And Priorities

1. **P0: Preflight and coverage gate**
   - Review `training_table_manifest.json`, `milestone6_feature_manifest.json`, and condition-normalization reports.
   - Produce `milestone7_coverage_report.json` with candidate coverage, out-of-catalog positive rates, evaluable groups, non-evaluable splits, and report-safety metadata.
   - Treat current eICU as non-evaluable for performance because positives are fully out of catalog.

2. **P1: Metric and reporting scaffold**
   - Implement metric helpers for AP, ROC-AUC, precision@k, recall@k, hit_rate@k, NDCG@k, and MRR@k.
   - Average ranking metrics over ranking groups with at least one in-catalog positive; separately report all group counts and positive-group coverage.
   - Add calibration outputs for probability-like scores: Brier score and 10-bin ECE where both classes are present.
   - Suppress subgroup metrics with fewer than 25 positive ranking groups.

3. **P2: Non-learned baselines**
   - Implement random, global popularity, and condition popularity.
   - Verify all popularity statistics are fit from MIMIC train only.
   - Use these baselines to validate scoring, ranking, tie-breaking, and report format before heavier models.

4. **P3: Learned baselines**
   - Implement linear and XGBoost training on MIMIC train sample.
   - Keep sampling memory-safe: select positives and deterministic weak
     negatives from narrow `patient_condition_medication` rows first, then join
     `patient_stay_features` only for selected rows.
   - Keep metric aggregation memory-safe: `append_metric_summaries` computes
     row-level and ranking metrics one `(baseline_name, source, split)` slice at
     a time so DuckDB only sorts a single window partition on the large
     final-mode score table. Per-slice results are unioned in Python and match
     the previous whole-table query exactly.
   - Use validation only for model comparison and any hyperparameter adjustment.
   - Save local ignored model artifacts and an aggregate manifest with feature list, parameters, seeds, software versions, and input artifact versions.

5. **P4: Sepsis/B3 and final evaluation**
   - After curated sepsis mappings are populated and harmonization/Milestone 6 artifacts are refreshed, filter headline metrics to `index_condition_token = condition:sepsis`.
   - Keep broad B1 per-condition metrics as context only.
   - Run final test evaluation only after validation-based choices are frozen.
   - Report eICU external validation only if it has in-catalog positives; otherwise report it as an external coverage failure, not model performance.

## Test Plan

- Add synthetic tests for:
  - ranking metrics with ties, multiple positives, no-positive groups, and `k` larger than group size;
  - AP/ROC-AUC returning null when a split has one class;
  - train-only popularity fitting;
  - deterministic random scores by seed;
  - learned baseline preprocessing with unknown categories;
  - no patient identifiers or row samples in reports;
  - test metrics blocked unless final/frozen mode is explicit.
- Add a small integration test using the existing Milestone 6 synthetic fixture.
- Verification commands:
  - `uv run pytest tests/test_config.py tests/test_milestone7_baselines.py`
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`

## Assumptions And Best Practices

- Use `uv` only.
- Use DuckDB for large joins, scoring tables, and aggregate metrics; use pandas/sklearn only for bounded sampled model training and batch scoring.
- Keep patient-level scores, models, and predictions under ignored `$DATASET_ROOT/processed/evaluation/`.
- Keep reports aggregate-only under `reports/`; never paste raw clinical rows, identifiers, note text, or row-level predictions.
- Do not claim clinical recommendation validity; labels are observed prescribing behavior and unobserved candidates are weak negatives.
- Do not pool MIMIC and eICU training in Milestone 7.
- Update `README.md`, `ARCHITECTURE.md`, `Documentation/DataFoundationRoadmap.md`, `WORKFLOWS.md`, `TESTING.md`, and `CHANGELOG.md` after implementation and protected-data review.
