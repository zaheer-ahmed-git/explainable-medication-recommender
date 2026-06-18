# Plan: Milestone 5 Harmonization With Extraction Prerequisite

## Summary

Build Milestone 5 as an executable data-foundation stage, including the missing Roadmap Milestone 4 source-specific extraction prerequisite. The result will be local, ignored, cohort-filtered Parquet extracts and harmonized source-tagged common schemas for demographics, conditions, medications, labs, vitals, allergies, interventions, and temporal events.

Current repo facts to respect: harmonization is not implemented; extraction modules are absent; broad adult cohort artifacts exist; prescriptions, labevents, inputevents, eICU medication, and APACHE result have newer passing profile/integrity artifacts; MIMIC `chartevents` must be treated as blocked unless a fresh integrity/profile run proves it is usable. No pooled training is enabled in this milestone.

## Key Changes

- Add prerequisite extraction modules:
  - `pipeline/mimic_extract.py`: cohort-filtered extracts from MIMIC patients/admissions/icustays, diagnoses, procedures, prescriptions, labevents, d_labitems, d_items, procedureevents, inputevents; defer `chartevents`, eMAR, pharmacy, and POE unless integrity gates pass.
  - `pipeline/eicu_extract.py`: cohort-filtered extracts from patient, diagnosis, lab, medication, infusionDrug, allergy, treatment, vitalPeriodic/vitalAperiodic, APACHE tables.
  - Each extractor writes only local ignored Parquet under `Dataset/processed/extracts/{source}/` plus aggregate manifests under `reports/`.

- Add `pipeline/harmonize.py`:
  - Read only extracted Parquet, never raw full tables.
  - Emit common source-tagged tables under `Dataset/processed/harmonized/`.
  - Preserve `source`, `source_version`, `patient_uid`, `encounter_uid`, `stay_uid`, original source IDs, original code/value/unit fields, extraction version, and mapping version.

- Implement RxNorm/ATC-first medication mapping:
  - Require mapping inputs under ignored `Dataset/mappings/medications/`, with expected files documented in config and manifest.
  - MIMIC primary mapping: `ndc -> rxcui -> ingredient/rxnorm_name -> atc_code/atc_level`.
  - eICU primary mapping: `drughiclseqno`/`gtc`/drug name to RxNorm or ATC via provided mapping tables.
  - Medication rows without RxNorm/ATC mapping remain in an unmapped report and are not silently dropped from source-specific extracts.

- Harmonized outputs:
  - `cohort_stays.parquet`: copied/enriched from current cohort artifacts with source-qualified IDs.
  - `conditions.parquet`: normalized ICD/eICU condition tokens plus original code/string provenance.
  - `medications.parquet`: RxNorm/ATC-normalized medication events, observed-order status, route/timing, and unmapped status.
  - `labs.parquet` and `vitals.parquet`: shared concepts for creatinine, lactate, WBC, platelets, sodium, potassium, glucose, HR, MAP, SpO2, temperature where available.
  - `allergies.parquet`, `interventions.parquet`, and `temporal_events.parquet`: source-tagged event layer for later feature construction.

- Add coverage and overlap reporting:
  - `reports/harmonization_manifest.json`
  - `reports/harmonization_coverage.json`
  - `reports/unmapped_concepts.json`
  - `notebooks/03_harmonization_and_overlap.ipynb` after core CLI reports exist.

## Interfaces And Contracts

- CLI commands:
  - `uv run python -m pipeline.mimic_extract`
  - `uv run python -m pipeline.eicu_extract`
  - `uv run python -m pipeline.harmonize`
  - Optional flags: `--dataset-root`, `--cohorts-root`, `--extracts-root`, `--harmonized-root`, `--reports-root`, `--mapping-root`.

- Config additions:
  - `EXTRACTS_ROOT = Dataset/processed/extracts`
  - `HARMONIZED_ROOT = Dataset/processed/harmonized`
  - `MAPPING_ROOT = Dataset/mappings`
  - `HARMONIZATION_VERSION`, `EXTRACTION_VERSION`, and explicit mapping file specs.

- Hard gates:
  - Extraction must join/filter to cohort IDs before materialization.
  - Any table with failed checksum/gzip/profile status is excluded unless a newer passing report is present.
  - RxNorm/ATC mapping resources must be present for medication harmonization; otherwise the harmonization command exits with a clear “mapping resources missing” manifest and nonzero status.
  - Unmapped condition, medication, lab, and vital concepts are reported with aggregate counts only.
  - Pooled MIMIC+eICU training stays disabled until reviewed coverage thresholds pass.

## Test Plan

- Add synthetic tests for:
  - Required-column validation and safe failure on missing mapping files.
  - Source-specific extraction filters rows to cohort IDs before writing.
  - Source-qualified IDs remain unique and provenance fields are present.
  - Medication mapping prefers RxNorm/ATC and reports unmapped tokens.
  - Lab/vital concept mapping preserves original units and reports incompatible units.
  - Harmonization never writes row samples, note text, or patient-level examples into reports.
  - Blocked source tables are skipped unless integrity/profile gates pass.

- Verification commands:
  - `uv run pytest tests/test_mimic_extract.py tests/test_eicu_extract.py tests/test_harmonize.py`
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`

## Documentation And Acceptance

- Update `README.md`, `ARCHITECTURE.md`, `Documentation/DataFoundationRoadmap.md`, `WORKFLOWS.md`, `TESTING.md`, and `CHANGELOG.md` to reflect implemented extraction/harmonization commands and current status.
- Acceptance criteria:
  - All new artifacts are written only under ignored `Dataset/processed/` or `reports/`.
  - Reports are aggregate-only and disclosure-conscious.
  - Every harmonized artifact records source, cohort version, extraction version, mapping version, and generated timestamp.
  - Mapping coverage is measurable by source and concept domain.
  - No claims are made that labels, features, baselines, Transformer-GNN models, or clinical recommendations are implemented.

## Assumptions And Risks

- User choice locked: include extraction prerequisite and use RxNorm/ATC-first medication normalization.
- Existing dependencies are sufficient; do not add packages unless a concrete mapping format requires it.
- If MIMIC `chartevents` remains blocked, v1 MIMIC vitals use only available dictionary/procedure/input-derived evidence and mark MIMIC charted vitals as unavailable.
- If medication mapping coverage is too low, complete the source-specific extracts and coverage reports but do not advance to candidate labels or pooled training.
- Rollback is simple: remove new pipeline modules/tests/docs changes from Git and delete ignored local artifacts under `Dataset/processed/extracts/`, `Dataset/processed/harmonized/`, and harmonization reports.
