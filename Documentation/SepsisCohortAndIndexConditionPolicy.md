# Sepsis Sub-Cohort and Index-Condition Policy

Status: **approved 2026-07-04 — implementation pending.** Approved choices:
**A1** (coded sepsis, reproducible now), index-condition policy **B1 → B3**
(all CCS/CCSR categories for the first Milestone 6 run, sepsis project-group
deep dive for Milestone 7 claims), with **A2 (Sepsis-3) planned after** the
`chartevents`/`inputevents` refresh. Code still reports
`sepsis_subcohort_status = not_implemented_pending_definition_approval` until
the implementation steps below land.

The roadmap requires approving "a source-specific sepsis definition,
terminology version, time window, and comparability strategy before coding"
(`Documentation/DataFoundationRoadmap.md`, Milestone 2 → Sepsis).

## Current State (grounding facts)

- Medication mapping coverage (2026-07-01 harmonization): MIMIC 83.5%
  (7.09M/8.48M rows), eICU 99.9%.
- Condition roll-up coverage: MIMIC rows are CCS/CCSR-mapped; eICU has
  ~431k `source_native_text` rows without a roll-up (~84% rolled up, right at
  the 85% target).
- `project_condition_groups.csv` and `eicu_diagnosis_text_condition_map.csv`
  exist only as **templates** — there is no curated sepsis
  `project_condition_token` yet, so the harmonizer's
  `project_condition_token` is currently NULL for all rows.
- The candidate catalog keys ranking groups on
  `COALESCE(project_condition_token, normalized_condition_token)`. With no
  project groups, this falls back to the CCS/CCSR category token today.
- `chartevents`/`inputevents` are not yet extracted; SOFA-based (Sepsis-3)
  organ-dysfunction scoring cannot be computed from current harmonized inputs.

## Part A — Sepsis Sub-Cohort Definition

### Option A1 — Coded sepsis (recommended v1, reproducible now)

Define the sepsis sub-cohort from billing/diagnosis codes and curated text,
with no chart-derived physiology. Reproducible from existing harmonized inputs.

- MIMIC-IV (ICD-9 and ICD-10 diagnoses):
  - ICD-9-CM: `995.91` (sepsis), `995.92` (severe sepsis), `785.52`
    (septic shock).
  - ICD-10-CM: `A41.*` (other sepsis), `A40.*` (streptococcal sepsis),
    `R65.20` (severe sepsis without septic shock), `R65.21` (septic shock).
  - Optionally roll up via the CCSR sepsis category (`SEP` family) for a
    terminology-versioned grouping.
- eICU-CRD:
  - `diagnosis` ICD-9/ICD-10 string codes matching the same code set, plus
    curated `diagnosisstring` / `apacheadmissiondx` text tokens
    (e.g. `sepsis`, `septic shock`, `severe sepsis`) via
    `eicu_diagnosis_text_condition_map.csv`.
- Shared token: `condition:sepsis` via `project_condition_groups.csv`
  (`match_type` in {`icd_code`, `text_token`}), preserving source-native code
  and text provenance.
- Time window: condition must be present for the stay (diagnosis is
  stay/admission-scoped). For ranking, the sepsis group defines the index
  condition; the Milestone 6 temporal contract (`t0` = admission,
  `t_pred = t0 + 24h`, label window `(24h, 48h]`) is unchanged.
- Terminology versions: ICD-9-CM, ICD-10-CM (MIMIC-IV v3.1), CCSR
  `condition-rollup-v1`, eICU-CRD v2.0 diagnosis strings.

Trade-off: simple, reproducible, and comparable across sources, but coded
sepsis under-captures early/uncoded sepsis and inherits billing bias.

### Option A2 — Sepsis-3 (deferred, requires upstream refresh)

Suspected infection (culture + antibiotic window) plus SOFA ≥ 2. More clinically
rigorous but requires `chartevents`/`inputevents` extraction and SOFA
derivation that do not exist yet. Recommend deferring until the
chartevents/inputevents refresh lands and a SOFA builder is implemented.

### Comparability strategy

Report sepsis stay counts per source and per mapping version; treat MIMIC as
development and eICU as external validation. Do not pool sources for training
until roll-up coverage and cross-source prevalence are reviewed. Record the
code set, terminology versions, and mapping version with every sepsis-filtered
artifact.

## Part B — Index-Condition Policy (ranking groups)

The index condition defines each ranking group in
`patient_condition_medication` (`ranking_group_id = stay_uid | index_condition_token`).

### Option B1 — All roll-up categories (current default)

Every mapped stay condition (CCS/CCSR category) forms a ranking group. Broadest
coverage; largest table (stay × conditions × top-50 candidates); useful for a
first plumbing run and aggregate manifest review. Weak clinical focus and
potentially very large row counts.

### Option B2 — Primary/admission diagnosis only

Restrict index conditions to the presenting problem: MIMIC `diagnoses_icd`
`seq_num = 1` (and/or admission diagnosis), eICU `apacheadmissiondx` /
primary diagnosis. Smaller, more interpretable, closer to a decision-time
question, but depends on diagnosis ordering semantics per source.

### Option B3 — Project groups only (sepsis-first deep dive)

Restrict index conditions to curated `project_condition_token` values
(sepsis first). Narrow, clinically focused, and comparable; requires the
curated `project_condition_groups.csv`. Recommended target for Milestone 7
evaluation claims.

### Recommendation

- Milestone 6 first materialization: keep **B1** (current code) to validate
  plumbing and produce aggregate manifests; inspect row counts and positive
  rates before scaling.
- Milestone 7 evaluation: adopt **B3** (sepsis deep dive) for headline claims,
  and report per-CCSR-category breakdowns from the B1 table for context.
- Pair with sepsis definition **A1** now; revisit **A2** after the
  chartevents/inputevents refresh.

## Approved Decisions (2026-07-04)

1. Sepsis definition: **A1** (coded sepsis, reproducible now). Revisit **A2**
   (Sepsis-3) after the `chartevents`/`inputevents` refresh enables SOFA.
2. Index-condition policy: **B1** (all CCS/CCSR categories) for the first
   Milestone 6 materialization, then **B3** (sepsis project-group deep dive)
   for Milestone 7 headline claims, with per-CCSR breakdowns for context.
3. Sequencing: run **B1** first so plumbing and aggregate manifests can be
   reviewed; curate the sepsis grouping and add the sub-cohort selector for the
   B3 deep dive.

## Implementation Steps (follow-up work)

1. Populate `Dataset/mappings/conditions/project_condition_groups.csv` with the
   approved A1 code/text set (ICD-9 `995.91`/`995.92`/`785.52`; ICD-10 `A40.*`,
   `A41.*`, `R65.20`, `R65.21`; text tokens `sepsis`, `severe sepsis`,
   `septic shock`) and the eICU diagnosis-text map, both flagged for clinical
   review.
2. Re-run `pipeline.harmonize` so `conditions.parquet` gains a non-null
   `project_condition_token = condition:sepsis` for matching rows.
3. Add a reproducible sepsis sub-cohort selector to `pipeline/cohort.py`,
   flipping `sepsis_subcohort_status` and recording the code set and
   terminology/mapping versions in the cohort manifest.
4. Milestone 7 evaluation filters ranking groups to `condition:sepsis` (B3)
   for headline metrics and reports per-CCSR-category breakdowns (B1) for
   context.
