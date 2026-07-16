# Milestone 6 Feature and Label Dictionary

## Scope

Milestone 6 converts Milestone 5 harmonized artifacts into local, ignored
feature and label artifacts for medication-ranking research. It does not train
baselines or recommendation models, build graph edges, pool MIMIC/eICU training,
or produce clinical recommendations.

All Parquet outputs below are patient-level derived artifacts and belong under
ignored `$DATASET_ROOT/processed/`. JSON manifests under `$PROJECT_HOME/reports/`
are aggregate-only and must not contain patient identifiers, source rows, note
text, or raw clinical examples.

## Temporal Contract

- Unit of analysis: ICU/unit stay.
- `t0`: ICU/unit admission.
- Feature cutoff: `t_pred = t0 + 24h`.
- Label window: medication starts with `24h < start_time <= 48h`.
- MIMIC events use absolute shifted timestamps relative to `stay_start_time`.
- eICU events use minute offsets divided by 60.
- Stays without observed coverage through 48h are excluded from the primary
  training table and counted in aggregate reports.
- Conditions are used to define ranking groups; untimed diagnosis rows are not
  included in default event-sequence features.

## Splits

`pipeline.features` assigns deterministic patient-level splits:

- MIMIC-IV patients: `train`, `validation`, or `test` using the configured seed.
- eICU patients: `external`.
- All stays for the same `patient_uid` must remain in one split.

## Feature Artifacts

### `cohort_decision_times.parquet`

Local path: `$DATASET_ROOT/processed/features/cohort_decision_times.parquet`

Key columns:

- `source`, `source_version`, `patient_uid`, `encounter_uid`, `stay_uid`
- `split`
- `t0_hours_from_admit`
- `prediction_time_hours_from_admit`
- `label_window_end_hours_from_admit`
- `t0_timestamp`, `prediction_timestamp`, `label_window_end_timestamp`
- `stay_end_hours_from_admit`, `los_hours`
- `eligibility_status`
- `primary_training_eligible`
- `cohort_version`, `harmonization_version`, `feature_version`, `split_version`

Eligibility values:

- `eligible_primary`
- `censored_before_prediction`
- `censored_before_label_window`
- `missing_observation_end`

### `patient_stay_features.parquet`

Local path: `$DATASET_ROOT/processed/features/patient_stay_features.parquet`

One row per stay. Default feature families:

- demographics and admission context;
- split and temporal bounds;
- aggregate pre-24h lab counts, numeric coverage, abnormal flags, and selected
  core lab summaries;
- aggregate pre-24h vital counts and selected core vital summaries;
- time-valid allergy/constraint presence;
- time-valid intervention counts.

Default core lab tokens:

- `creatinine`
- `lactate`
- `wbc`
- `platelets`
- `sodium`
- `potassium`
- `glucose`

Default core vital tokens:

- `heart_rate`
- `mean_arterial_pressure`
- `spo2`
- `temperature`
- `respiratory_rate`

Medication-history features are excluded by default to reduce target-proxy
leakage risk.

Missingness handling:

- core lab/vital observation flags and count columns are emitted directly in
  `patient_stay_features.parquet`;
- imputation and scaling are not baked into this raw feature table; they are
  fit from MIMIC train rows only by `pipeline.preprocessing` and saved as a
  local ignored artifact.

### Optional Phase 8 P0 feature set

`pipeline.features --feature-set phase8_p0` writes `temporal-features-v2` into
caller-selected roots, usually
`$DATASET_ROOT/processed/phase8_p0/features/`. The baseline default remains
`temporal-features-v1`.

Additional `patient_stay_features.parquet` columns:

- condition presence columns named `condition_{safe_token}_present_24h`, fit
  from the top MIMIC train condition tokens by distinct stay frequency. Token
  precedence is `normalized_condition_token`, then `project_condition_token`;
  validation/test/eICU-only tokens do not create columns.
- for each default core lab and vital token:
  `*_first_24h`, `*_last_24h`, `*_delta_24h`, `*_slope_24h`, and
  `*_hours_since_last_24h`, using only events with
  `0h <= event_time_hours_from_admit <= 24h`;
- explicit complementary `*_missing_24h` indicators for the same core lab and
  vital tokens.

Phase 8 P0 manifests add aggregate-only `condition_vocabulary_size`,
`condition_columns_added`, `trend_columns_added`,
`missingness_columns_added`, `feature_column_counts_by_family`, and
`condition_oov_counts`. They must not list patient rows or out-of-vocabulary
condition token values.

### `event_sequences.parquet`

Local path: `$DATASET_ROOT/processed/features/event_sequences.parquet`

Pre-decision events only, with `event_time_hours_from_admit <= 24h`.

Default behavior excludes medication events. A CLI flag can include
pre-decision medication events for reviewed experiments, but this should be
treated as a leakage-sensitive decision.

At protected-data scale, the builder materializes reduced pre-decision events
once, windows them in stay-hash batches, then combines the batches into this
single canonical Parquet file. Tune the batch count with
`--event-sequence-batches` or `EVENT_SEQUENCE_BATCHES` in the OAR wrappers.

Key columns:

- identifiers and `split`
- `event_sequence_position`
- `event_type`, `source_domain`, `source_table`, `source_event_id`
- `event_time_hours_from_admit`
- `event_token`, `source_code`, `source_text`
- `value_numeric`, `value_text`, `unit`, `normalized_unit`
- provenance fields

## Training Artifacts

### `split_manifest.parquet`

Local path: `$DATASET_ROOT/processed/training/split_manifest.parquet`

One row per source-qualified patient and split, including stay count and split
provenance.

### `candidate_catalog.parquet`

Local path: `$DATASET_ROOT/processed/training/candidate_catalog.parquet`

Condition-specific candidates learned only from MIMIC train positives.

Candidate rules:

- condition token: `COALESCE(project_condition_token, normalized_condition_token)`;
- default medication token strategy: `rxnorm_or_atc`
  (`rxnorm:{rxcui}` when available, else `atc:{ATC3}`);
- coverage-sensitivity medication token strategy: `atc3_or_rxnorm`
  (`atc:{ATC3}` when available, else `rxnorm:{rxcui}`), intended for
  out-of-catalog positive and cross-source overlap analysis;
- unmapped conditions and medications are excluded from candidate construction
  and counted in aggregate reports;
- default cap: top 50 candidates per condition by train positive stay count.

### `patient_condition_medication.parquet`

Local path:
`$DATASET_ROOT/processed/training/patient_condition_medication.parquet`

One row per eligible `stay + condition + candidate medication`.

Key columns:

- identifiers and `split`
- `index_condition_token`, `index_condition_name`
- `candidate_medication_token`, `candidate_medication_name`, `candidate_rank`
- `ranking_group_id`
- `label_prescribed`
- `label_first_observed_hours_from_admit`
- `label_event_count`
- `label_semantics`
- temporal and provenance fields

`label_prescribed = true` means at least one matching medication start was
observed in the label window. `false` means the candidate was not observed in
that window and is only a weak observational negative.

## Preprocessing Artifacts

### `train_preprocessing_sample.parquet`

Local path:
`$DATASET_ROOT/processed/training/preprocessing/train_preprocessing_sample.parquet`

Deterministic MIMIC train-only sample used to fit tabular preprocessing. The
sample includes all train positives and a deterministic weak-negative sample
using the configured seed.

### `train_fitted_preprocessor.joblib`

Local path:
`$DATASET_ROOT/processed/training/preprocessing/train_fitted_preprocessor.joblib`

Contains the fitted sklearn preprocessing object:

- numeric median imputation and scaling;
- categorical constant imputation and one-hot encoding;
- categorical vocabularies learned from MIMIC train rows only;
- feature-column metadata and fit-scope metadata.

The fitted object is a local derived artifact and may contain source concept or
site vocabulary values. It is ignored and must not be committed. The public JSON
manifest reports category counts only, not vocabulary values.

## Reports

Aggregate-only reports:

- `reports/milestone6_feature_manifest.json`
- `reports/training_table_manifest.json`
- `reports/preprocessing_manifest.json`
- Phase 8 P0 isolated reports, when that ablation is run:
  `reports/phase8_p0_milestone6_feature_manifest.json`,
  `reports/phase8_p0_training_table_manifest.json`, and
  `reports/phase8_p0_preprocessing_manifest.json`

These reports include artifact paths, versions, parameter values, row counts,
split counts, censoring counts, event-exclusion counts, candidate counts,
out-of-catalog positive counts, coverage-loss counts, train-fit preprocessing
counts, and category cardinalities. They must not include patient identifiers
or row samples.

## Commands

```bash
uv run python -m pipeline.features
uv run python -m pipeline.build_training_table
uv run python -m pipeline.preprocessing
# Optional coverage sensitivity:
uv run python -m pipeline.build_training_table --candidate-token-strategy atc3_or_rxnorm
# Optional Phase 8 P0 isolated feature build:
uv run python -m pipeline.features --feature-set phase8_p0 \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json"
```

Run full protected-data materialization only after Milestone 5 coverage and
mapping gates are reviewed. Use OAR for heavy Calculco runs.

## Related planning

Phase 8 P0 extensions and hybrid branch boundaries are recorded in
`Documentation/HybridModelFeatureStrategy.md`. They are ablation features until
the isolated protected-data reruns and feature-gate review justify promotion.
