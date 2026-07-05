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
- medication token: `rxnorm:{rxcui}` when available, else `atc:{atc_code}`;
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

## Reports

Aggregate-only reports:

- `reports/milestone6_feature_manifest.json`
- `reports/training_table_manifest.json`

These reports include artifact paths, versions, parameter values, row counts,
split counts, censoring counts, event-exclusion counts, candidate counts,
out-of-catalog positive counts, and coverage-loss counts. They must not include
patient identifiers or row samples.

## Commands

```bash
uv run python -m pipeline.features
uv run python -m pipeline.build_training_table
```

Run full protected-data materialization only after Milestone 5 coverage and
mapping gates are reviewed. Use OAR for heavy Calculco runs.
