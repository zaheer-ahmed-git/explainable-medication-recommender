# Milestone 6 Execution Plan: Temporal Features and Labels

## Summary
Milestone 6 converts the completed Milestone 5 harmonized artifacts into model-ready, leakage-controlled feature and label artifacts. It does not train baselines, build graph edges, run Transformer-GNN models, enable pooled MIMIC/eICU training, or produce clinical recommendations.

Primary inputs are the harmonized Parquet tables under `Dataset/processed/harmonized/`. Primary outputs are local ignored artifacts for temporal contracts, patient splits, stay features, event sequences, candidate catalogs, and `patient_condition_medication`, plus aggregate-only manifests under `reports/`.

Before any protected-data run, confirm Milestone 5 coverage has been reviewed, medication mapping passed the hard gate, condition normalization coverage is acceptable for the intended analysis, and `PROJECT_HOME`, `DATASET_ROOT`, and `WORK_SCRATCH` are exported on Calculco.

## Implementation Status

Implemented with synthetic tests; protected-data materialization pending reviewed Milestone 5 coverage gates.

- `pipeline/features.py` builds `cohort_decision_times`, `patient_stay_features`, and `event_sequences`.
- `pipeline/build_training_table.py` builds `split_manifest`, `candidate_catalog`, and `patient_condition_medication`.
- `pipeline/config.py` defines `FEATURES_ROOT`, `TRAINING_ROOT`, `FEATURE_VERSION`, `LABEL_VERSION`, `SPLIT_VERSION`, and `DEFAULT_MODELING_PARAMETERS`.
- `tests/test_features.py` and `tests/test_build_training_table.py` cover the temporal, split, candidate, and label contracts on synthetic fixtures.
- OAR wrappers `scripts/calculco/features.sh`, `scripts/calculco/build_training_table.sh`, and `scripts/calculco/milestone6.sh` run the protected-data materialization on Calculco.

This document is the execution plan and rationale. The canonical artifact
schemas, column lists, and leakage caveats live in
`Documentation/Milestone6FeatureLabelDictionary.md`; current milestone status
lives in `Documentation/DataFoundationRoadmap.md`. Keep schema detail in the
dictionary rather than duplicating it here.

## Key Interfaces And Artifacts
- Add config constants in `pipeline/config.py`: `FEATURES_ROOT`, `TRAINING_ROOT`, `FEATURE_VERSION`, `LABEL_VERSION`, `SPLIT_VERSION`, and reuse `DEFAULT_MODELING_PARAMETERS` defaults: top 50 candidates, `t_pred = t0 + 24h`, 24h label window, seed `20260617`.
- Add `pipeline/features.py` with `FeatureBuildConfig` and `build_feature_artifacts(config)`. CLI: `uv run python -m pipeline.features`.
  `event_sequences` uses a staged, stay-hash-batched windowing path
  (`--event-sequence-batches`, default 8) so large `temporal_events` inputs do
  not require one global `ROW_NUMBER()` over all pre-decision events.
- Add `pipeline/build_training_table.py` with `TrainingTableBuildConfig` and `build_training_artifacts(config)`. CLI: `uv run python -m pipeline.build_training_table`.
- Write ignored artifacts:
  - `Dataset/processed/features/cohort_decision_times.parquet`
  - `Dataset/processed/features/patient_stay_features.parquet`
  - `Dataset/processed/features/event_sequences.parquet`
  - `Dataset/processed/training/split_manifest.parquet`
  - `Dataset/processed/training/candidate_catalog.parquet`
  - `Dataset/processed/training/patient_condition_medication.parquet`
- Write aggregate-only reports:
  - `reports/milestone6_feature_manifest.json`
  - `reports/training_table_manifest.json`
  - no patient IDs, row samples, note text, or raw source values.
- Add a safe tracked data dictionary, e.g. `Documentation/Milestone6FeatureLabelDictionary.md`, describing artifact schemas, temporal rules, excluded leakage features, and observational-label caveats.

## Phased Execution
1. **P0 readiness gate**
   Verify harmonization manifest includes all required tables. Review aggregate coverage/unmapped reports. If refreshed MIMIC `chartevents`/`inputevents` extraction is needed, complete that before the real protected-data run; implementation can still proceed with synthetic tests.

2. **P1 temporal contract**
   Define `t0` as ICU/unit admission. For MIMIC use absolute stay/event timestamps; for eICU use minute offsets from ICU admission. Normalize both to `event_time_hours_from_admit`. Keep features with time `<= 24h`; labels are medication starts with `24h < start_time <= 48h`.

3. **P1 censoring and eligibility**
   Primary training rows require known stay coverage through 48h. Stays ending before `t_pred` or before full label-window observation are excluded from the primary table and counted in aggregate exclusion reports. Do not silently treat censored unobserved candidates as negatives.

4. **P1 patient split**
   Split MIMIC patients deterministically 80/10/10 into train/validation/test using the configured seed. Assign all eICU patients to `external` by default. Enforce one patient, one split.

5. **P2 feature construction**
   Build one row per stay in `patient_stay_features`: demographics, admission/ICU context, source, split, temporal bounds, lab/vital summaries before `t_pred`, missingness indicators, prior intervention counts, and conservative allergy/constraint flags only when time-valid. Exclude outcomes, discharge-only fields, full-corpus popularity, future events, and candidate-specific prior-medication leakage from default features.

6. **P2 event sequences**
   Build `event_sequences` from time-valid pre-decision events only. Preserve event type, token, numeric/text value, normalized unit, source domain, provenance, and `event_time_hours_from_admit`. Use domain tables directly when normalized condition or medication fields are needed. For protected-data scale, first stage the reduced pre-decision events in one scan, apply `ROW_NUMBER()` in stay-hash batches, then combine the part files into the canonical single `event_sequences.parquet`.

7. **P2 candidate catalog**
   Build condition-specific candidate catalogs from MIMIC train positives only. Use `index_condition_token = COALESCE(project_condition_token, normalized_condition_token)`. Use canonical medication token `rxnorm:{rxcui}` when available, else `atc:{atc_code}`. Exclude unmapped condition/medication rows from candidates and report aggregate coverage loss.

8. **P2 labels and negatives**
   Create one row per eligible `stay + condition + candidate medication`. Positives are observed medication starts in the label window; repeated or simultaneous starts collapse to the earliest positive event. Negatives are catalog candidates not observed in that window and must be documented as weak observational negatives, not clinical non-indications. Report out-of-catalog positives.

9. **P3 documentation and workflow updates**
   Update README/architecture/roadmap/workflows/testing/changelog only after implementation, clearly marking Milestone 6 as implemented and baselines/models as still planned.

## Test Plan And Acceptance Gates
- Add `tests/test_features.py`, `tests/test_build_training_table.py`, and focused config tests.
- Required synthetic cases: MIMIC absolute timestamps, eICU offsets, boundary times at 24h/48h, repeated medications, censored stays, missing times, train-only candidates, out-of-catalog positives, deterministic splits, and aggregate-only reports.
- Verification commands:
  - `uv run pytest tests/test_config.py tests/test_features.py tests/test_build_training_table.py`
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
- Milestone exit gate: temporal cutoff tests pass, patient split integrity passes, candidate catalogs are train-only, default features exclude leakage-prone fields, reports contain no patient-level data, and every artifact records cohort/feature/label/split/mapping/provenance versions.

## Assumptions And Best Practices
- `Documentation/CodexPLAN.md` is the requested `codexplan.md`.
- No new dependencies are needed.
- Use DuckDB with the existing configured spill/memory/thread controls for large artifact builds.
- Use `uv` exclusively.
- Keep protected-data artifacts under ignored `Dataset/processed/`; keep reports aggregate-only.
- Run heavy protected-data materialization via OAR, not interactively on the login node.
- Preserve the project claim boundary: observed prescriptions are historical labels, not optimal treatments.
