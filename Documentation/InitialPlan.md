# Goal and Scope

Build a deep, justified understanding of **MIMIC-IV v3.1** and **eICU-CRD v2.0**, and create a clean data foundation for the hybrid **Transformer + GNN medication recommender** described in:

- `Documentation/ResearchDetail.md`
- `Docume2ntation/PosterPresentationGuide.md`

## Decisions Locked from Clarification

### Cohort
- ICU-stay-level
- All conditions
- Candidate medications limited to top-N most-prescribed per condition
- Sepsis used as the worked deep-dive / first end-to-end iteration

### Compute
- DuckDB + chunked streaming over gzipped CSVs for multi-GB tables
- In-memory pandas only on cohort-filtered subsets

### Deliverables
- Runnable scripts
- EDA notebooks
- Written roadmap document

### Modeling Unit
One row per patient + condition + medication candidate with:
- `label_prescribed`
- Patient-level split
- Leakage controls

---

# Key Local Data Findings

- MIMIC-IV-Note v2.2 is not present locally.
- Only eICU notes are available.
- Large tables must be queried, not loaded into memory.
- Raw PhysioNet data must never be committed or redistributed.

---

# Proposed Repository Structure

```text
pipeline/
notebooks/
Documentation/
reports/
Dataset/processed/
DepreciatedCode/
```

## Pipeline Modules

- `pipeline/config.py` - paths, cohort params, candidate params, big-table list, RNG seed.
- `pipeline/io_utils.py` - DuckDB connection helpers, chunked gzip readers, cohort-ID filtering, parquet caching to Dataset/processed/cache/.
- `pipeline/cohort.py` - define the ICU cohort for each source (see below).
- `pipeline/profile_tables.py` - schema + quality profiling for every table (row counts, dtypes, null %, cardinality, duplicates, key integrity).
- `pipeline/mimic_extract.py` / pipeline/eicu_extract.py - cohort-scoped extraction of demographics, diagnoses, labs, vitals, meds, allergies, severity.
- `pipeline/harmonize.py` - map both sources into one unified schema + shared vocabularies (see Combining section).
- `pipeline/features.py` - feature engineering (aggregations, trends, abnormality flags, comorbidity indices).
- `pipeline/build_training_table.py` - candidate generation, labels, temporal cutoff, patient-level split.

## Notebooks

- `01_schema_quality.ipynb`
- `02_distributions_correlations.ipynb`
- `03_harmonization_and_overlap.ipynb`
- `04_graph_suitability.ipynb`
- `05_feature_selection.ipynb`

## Documentation

- `Documentation/DataFoundationRoadmap.md`

---

# Cohort Definition

## MIMIC-IV

- Adult ICU stays
- First ICU stay per admission
- Derived from `icustays`, `patients`, and `admissions`

## eICU

- One row per ICU unit stay
- Derived from `patient.csv.gz`

## Sepsis Sub-Cohort

Used as the first end-to-end deep dive and model iteration.

---

# Phased Workflow
# Phase 0 - Setup and cohort

Add deps, write `config.py`/`io_utils.py`/`cohort.py`, materialize cohort ID lists and a cohort manifest (counts, date ranges) for both sources.

# Phase 1 - Structure, schema, data quality (items 1-2)

`profile_tables.py` + `01_schema_quality.ipynb`: per-table row counts, column dtypes, primary/foreign-key integrity (e.g., every `stay_id`/`patientunitstayid` resolves), null/missing %, duplicate rows and duplicate keys, value-range sanity (impossible ages, negative LOS, lab unit inconsistencies), missingno matrices. Output `reports/quality_profile.json`.

# Phase 2 - Statistical analysis and profiling (item 3)

Cohort-level descriptive stats: demographics, diagnosis prevalence, top medications, lab/vital summary stats with outlier detection (IQR + clinical plausibility bounds), ydata-profiling reports for the cohort-filtered extracts.

# Phase 3 - Visualizations: distributions, correlations, trends (item 4)

`02_distributions_correlations.ipynb`: distributions (age, LOS, labs, vitals), medication/diagnosis frequency Pareto plots, lab-feature correlation heatmaps, time trends (length-of-stay, prescribing over time), MIMIC-vs-eICU side-by-side comparisons. Figures saved to `reports/figures/`.

# Phase 4 - Feature engineering (item 6)

`features.py`: per-stay aggregations (lab mean/min/max/last, abnormal-flag counts, trend/slope), vital summaries, comorbidity counts + Charlson/Elixhauser-style index from ICD, severity (APACHE for eICU; admission/ICU context for MIMIC), allergy flags (eICU explicit), prior-intervention counts. Document each engineered feature and its rationale.

# Phase 5 - Feature importance and relevance (item 5)

`05_feature_selection.ipynb`: build the candidate training table, then rank feature relevance via mutual information, a quick XGBoost importance pass (reusing conventions from `DepreciatedCode/xgboost_medication_ranker.py`), and correlation-with-target. Output a ranked feature table.

# Phase 6 - Preprocessing (item 7)

`build_training_table.py` + sklearn-style transforms: cleaning, categorical encoding, numeric imputation, scaling/normalization, and class-imbalance handling for the positive/negative candidate labels (class weights / negative sampling). Patient-level train/valid/test split (deterministic hash like the deprecated `stable_patient_split`).

# Phase 7 - Bias, leakage, limitations (item 8)

Document and enforce:

- temporal cutoff (only data before prescription time used as features)
- patient-level split (no patient across splits)
- exclusion of outcome/history/popularity leakage features by default
- single-center vs multi-center bias (MIMIC one site, eICU 208)
- demographic representation bias
- missing-not-at-random labs
- and the missing MIMIC-Note limitation

# Phase 8 - Component routing + graph suitability (items 9-10)

`04_graph_suitability.ipynb` + roadmap section deciding which features feed each component:

## Transformer branch

Per-stay tabular/sequential features (labs, vitals, demographics, diagnosis flags, prior interventions) as a feature/token sequence; clinical text (eICU notes) as optional text-encoder branch.

## GNN branch

Heterogeneous graph nodes (patient, diagnosis/ICD, medication, lab/abnormality) with edges from co-occurrence (patient-diagnosis, diagnosis-medication, patient-medication) and similarity; quantify graph density, degree distribution, connectivity, and co-occurrence signal strength to justify GNN suitability.

## Fusion layer

Concatenation/attention over the Transformer patient-context embedding and GNN entity embeddings; defines inputs and output ranking head.

# Phase 9 - Roadmap doc with justified feature selection (items 11-12)

`Documentation/DataFoundationRoadmap.md`: synthesizes all figures/tables into the final feature-selection strategy, the MIMIC+eICU combination plan, and a concrete build sequence for the hybrid module.

# Combining MIMIC-IV and eICU (item 12)

Harmonize, do not blindly row-concat. Map both into one unified schema with a shared source tag and shared vocabularies, then support two usage modes:

1. MIMIC for development + eICU as external validation
2. Optional pooled training on the harmonized union

## Datasets

- MIMIC-IV v3.1
- eICU-CRD v2.0

## Extraction and Harmonization Components

- `mimic_extract.py`
- `eicu_extract.py`
- `harmonize.py`: unified schema + shared vocab
- Unified stay-level tables (source-tagged)
- `build_training_table.py`: candidates + labels + split
- Transformer + GNN + Fusion

## MIMIC Sources

- `icu/icustays`
- `hosp/patients + admissions`
- `diagnoses_icd + d_icd_diagnoses`
- `labevents + d_labitems`
- `prescriptions / pharmacy / emar`

## eICU Sources

- `patient`
- `diagnosis`
- `lab`
- `medication / infusionDrug`
- `allergy`

## Harmonization mapping (high level)

### IDs

MIMIC `subject_id`/`hadm_id`/`stay_id` and eICU `uniquepid`/`patientunitstayid` map to unified `patient_uid` + `stay_uid` + `source`.

### Conditions

MIMIC ICD-9/10 vs eICU diagnosis strings/ICD - normalize to a shared condition vocabulary (ICD-10 roll-up or condition tokens), reusing snake_case normalization from the deprecated pipeline.

### Medications

MIMIC drug names vs eICU drug names - normalize to ingredient-level tokens (and ideally RxNorm/ATC-style grouping) so candidate catalogs align across sources.

### Labs/vitals

Map to a shared concept set (e.g., creatinine, lactate, WBC, HR, MAP) with unit harmonization.

Schema differences and any unmapped concepts are logged; cross-dataset coverage is reported so we know which shared features are usable for pooled training vs validation.

# Validation of the Approach

- Reproduce cohort counts against documented dataset scales (MIMIC ~94k ICU stays; eICU ~200k unit stays) as a sanity check.
- Confirm no patient crosses splits.
- Confirm temporal cutoff removes future-dated features.
- Confirm shared-vocabulary coverage is high enough on the top-N candidate medications before any pooled training.

# Out of Scope (for this EDA phase)

Training the actual Transformer/GNN models, knowledge-graph construction, LIME/explainability, and the LLM conversational layer. This phase delivers the data foundation and the justified feature/architecture routing that those steps build on.

---

- Add deps (`duckdb`, `pyarrow`, `ydata-profiling`, `missingno`, `networkx`) via uv; create `pipeline/config.py`, `io_utils.py`, and `cohort.py` with DuckDB/chunked readers and ICU cohort ID materialization for both sources.

- Implement `profile_tables.py` and `01_schema_quality.ipynb`: schema, key integrity, null %, duplicates, range sanity, missingno; output `reports/quality_profile.json`.

- Cohort descriptive stats + outlier detection + ydata-profiling; `02_distributions_correlations.ipynb` with distributions, frequency Paretos, correlation heatmaps, trends, and MIMIC-vs-eICU comparisons saved to `reports/figures/`.

- Implement `mimic_extract.py`, `eicu_extract.py`, and `harmonize.py`: cohort-scoped extraction and mapping into one unified source-tagged schema with shared condition/medication/lab vocabularies; `03_harmonization_and_overlap.ipynb` reports cross-dataset coverage.

- Implement `features.py`: per-stay lab/vital aggregations and trends, abnormality flags, comorbidity/Charlson index, severity (APACHE/admission context), allergy and prior-intervention features, each documented with rationale.

- Implement `build_training_table.py`: candidate generation (top-N meds per condition), `label_prescribed`, temporal cutoff, encoding/imputation/scaling, class-imbalance handling, and deterministic patient-level split.

- `05_feature_selection.ipynb`: mutual information, XGBoost importance pass, and target correlation to produce a ranked feature relevance table.

- `04_graph_suitability.ipynb`: build heterogeneous co-occurrence graph, quantify density/degree/connectivity, and decide Transformer vs GNN vs fusion feature routing.

- Document biases, leakage risks, and limitations (temporal cutoff, patient-level split, single- vs multi-center, missing MIMIC-Note, MNAR labs) and enforce leakage controls in the pipeline.

- Write `Documentation/DataFoundationRoadmap.md` synthesizing figures/tables into the justified feature-selection strategy, MIMIC+eICU combination plan, and the hybrid-module build sequence.