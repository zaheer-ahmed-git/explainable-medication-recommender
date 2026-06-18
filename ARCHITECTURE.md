# Architecture

## Purpose

This document separates the repository's current implementation state from its
target research architecture.

## Current State

As of 2026-06-17, the active repository contains research documents,
configuration, local licensed datasets, an ignored synthetic prototype,
metadata-only source inventory, and adult ICU/unit-stay cohort materialization.
The active `pipeline/` provides configuration, safe path/header inspection,
bounded DuckDB CSV reads, source-inventory CLI, cohort CLI, and aggregate
source-table quality profiling plus aggregate EDA briefing synthesis.
Harmonization, analysis notebooks, feature tables, labels, and recommendation
models are not yet implemented in the active working tree.

The legacy prototype demonstrates useful conventions such as:

- condition-specific medication candidate generation;
- deterministic patient-level train, validation, and test splits;
- baseline linear and XGBoost scoring;
- ranking metrics such as precision, recall, hit rate, NDCG, and MRR at K; and
- default exclusion of medication-history, outcome, and popularity features.

These conventions are references, not validated components of the target
clinical-data pipeline.

## Target Boundaries

The system is divided into six boundaries:

1. **Source ingestion:** bounded access to MIMIC-IV, MIMIC-IV-Note, and eICU.
2. **Cohort and harmonization:** source-specific cohorts mapped into a common,
   provenance-preserving schema.
3. **Feature and label construction:** pre-decision patient context, medication
   candidates, observed labels, temporal cutoffs, and patient-level splits.
4. **Recommendation:** baseline models followed by Transformer and GNN branches
   with a fusion ranking head.
5. **Grounded explanation:** model attribution, graph paths, rule results,
   uncertainty, contradiction handling, and provenance.
6. **Conversational review:** structured extraction and clinician-facing
   presentation. The LLM does not independently prescribe medication.

## Planned Repository Structure

```text
pipeline/
  config.py
  io_utils.py
  source_inventory.py
  cohort.py
  profile_tables.py
  eda_summary.py
  mimic_extract.py
  eicu_extract.py
  harmonize.py
  features.py
  build_training_table.py
tests/
notebooks/
reports/
Dataset/processed/
Documentation/
```

`Dataset/`, `reports/`, and model artifacts remain ignored. Small synthetic test
fixtures may be versioned under `tests/fixtures/`.

## Data Flow

```text
Licensed source tables
        |
        v
Source-specific cohort IDs and manifests
        |
        v
Bounded extraction by source
        |
        v
Unified stay, diagnosis, medication, lab, vital, and note concepts
        |
        v
Pre-decision features + candidate medications + observed labels
        |
        v
Patient-level train/validation/test partitions
        |
        v
Baselines -> Transformer branch + GNN branch -> fusion ranker
        |
        v
Attribution + graph evidence + rules + uncertainty + provenance
        |
        v
Clinician-reviewable ranked options
```

## Source Architecture

### MIMIC-IV

Use `subject_id`, `hadm_id`, and `stay_id` as distinct source keys. Relevant
domains include admissions, patients, ICU stays, diagnoses, laboratory events,
prescriptions, pharmacy, medication administration, and ICU events.

### MIMIC-IV-Note

Discharge and radiology notes may support clinical-text experiments. Text use
requires the same restricted-data controls as structured tables. Notes must not
be pasted into prompts, fixtures, logs, or public artifacts.

### eICU-CRD

Use `uniquepid` and `patientunitstayid` carefully because a patient may have
multiple unit stays. Relevant domains include patient, diagnosis, lab,
medication, infusion, allergy, APACHE, treatment, notes, and vitals.

## Unified Identifiers

Derived tables should use source-qualified identifiers:

- `source`
- `patient_uid`
- `encounter_uid`
- `stay_uid`

Never assume source-local integer identifiers are globally unique. Preserve the
original identifiers in restricted derived artifacts for traceability.

## Temporal Contract

Every model row must define:

- an index condition or decision point;
- a feature window ending before the target medication decision;
- a label window;
- allowable history;
- excluded future information; and
- the rule used to resolve simultaneous or repeated prescriptions.

Outcome variables recorded after treatment are evaluation context unless the
experiment defines a valid historical cutoff.

## Recommendation Contract

The primary task is ranking condition-appropriate medication candidates for a
patient or stay. Candidate catalogs must be learned from training data only.
Observed prescriptions are positive historical labels; unobserved candidates
are not automatically true negatives.

Start with transparent baselines before the hybrid model:

- random and popularity baselines;
- condition-frequency baseline;
- linear classifier/ranker;
- gradient-boosted tree ranker.

The Transformer-GNN model is justified only if it improves held-out ranking,
safety, calibration, and external validation over these baselines.

## Hybrid Model

### Transformer Branch

Models contextual and temporal interactions among demographics, diagnoses,
laboratories, vitals, prior interventions, and optional note representations.

### GNN Branch

Models heterogeneous relations among patients or stays, diagnoses,
medications, laboratory concepts, and curated medical knowledge. Graph
construction must avoid test-set and future-event leakage.

### Fusion

Combines patient-context and graph-aware embeddings to score medication
candidates. Fusion choices and ablations must be recorded.

## Explanation Boundary

The explanation layer consumes evidence produced by the recommendation and
knowledge layers:

- local feature or token attribution;
- graph substructures and evidence paths;
- rule and contraindication results;
- model scores and calibration;
- missing-data and conflict indicators; and
- provenance records.

The LLM may structure inputs and verbalize evidence. It must not invent
unsupported clinical claims or substitute narrative confidence for evidence.

## Safety and Governance

- No raw patient records enter Git history.
- No clinical recommendation is presented without uncertainty and limitations.
- Hard safety rules may flag or block a high model score.
- Conflicting evidence must be surfaced rather than hidden.
- Every reported result must identify cohort, split, feature window, label
  definition, source, model version, and evaluation code version.

## Open Decisions

- Exact sepsis cohort definition and coding system.
- Shared condition vocabulary and roll-up level.
- Medication ingredient normalization and RxNorm/ATC mapping strategy.
- Prescription decision time and label window.
- Graph node and edge definitions.
- Transformer input representation.
- Rule-source curation and versioning.
- Human evaluation protocol for explanation usefulness.

Resolve these decisions in reviewed plans before implementation.
