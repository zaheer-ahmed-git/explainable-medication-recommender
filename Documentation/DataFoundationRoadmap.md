# Data Foundation Roadmap

## Purpose

This roadmap turns the research architecture into a sequence of verifiable
data and modeling milestones. It supersedes status assumptions in older plans
when they conflict with the current working tree.

Last reviewed: 2026-06-17.

## Current Baseline

Available locally under the ignored `Dataset/` directory:

- MIMIC-IV v3.1 hospital and ICU tables;
- MIMIC-IV-Note v2.2 discharge and radiology notes; and
- eICU-CRD v2.0 tables.

The source files include multi-gigabyte compressed CSV tables. They require
DuckDB projection/filtering or chunked streaming.

The active repository contains source-inventory, adult ICU/unit-stay cohort
materialization, aggregate source-table quality profiling, and aggregate EDA
briefing synthesis code with synthetic tests. EDA notebooks, harmonization,
feature construction, labels, graph artifacts, and models are not yet
implemented. The ignored
`DepreciatedCode/` prototype supplies historical conventions for candidate
generation, patient-level splitting, baseline ranking, and ranking metrics.

## Locked Research Direction

- Unit of early analysis: ICU stay, with source-qualified patient and encounter
  identifiers.
- Initial deep-dive condition: sepsis, after a reproducible cohort definition
  is approved.
- Candidate task: rank condition-appropriate medications for a patient/stay.
- Preferred validation: MIMIC-IV development and eICU external validation.
- Optional pooled training: only after harmonization coverage and semantics
  pass explicit gates.
- Compute: DuckDB and chunked reads for source tables; pandas only for bounded
  extracts.
- Raw data: local, licensed, ignored, and non-redistributable.

## Target Repository Shape

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
  fixtures/
notebooks/
  01_schema_quality.ipynb
  02_distributions_correlations.ipynb
  03_harmonization_and_overlap.ipynb
  04_graph_suitability.ipynb
  05_feature_selection.ipynb
reports/
Dataset/processed/
```

## Cross-Cutting Gates

Every milestone must satisfy:

- no raw or patient-level data in Git, logs, prompts, or public artifacts;
- source-qualified identifiers and provenance;
- bounded processing for large tables;
- documented cohort and temporal contracts;
- patient-level split integrity;
- training-only fitting of candidates, vocabularies, and transforms;
- synthetic tests;
- reproducible configuration and manifests; and
- explicit current-versus-planned status in documentation.

## Milestone 0: Governance and Reproducibility

Status: completed for the initial repository documentation layer.

Deliverables:

- durable project instructions;
- architecture, workflow, testing, review, security, and contribution docs;
- Cursor and Codex project configuration;
- reusable verification and data-safety skills;
- ignored paths for licensed data and generated artifacts.

Exit gate:

- documentation and configuration parse successfully;
- commands and paths match the current tree.

## Milestone 1: Pipeline Skeleton and Source Inventory

Status: implemented for initial metadata-only source inventory.

Deliverables:

- `pipeline/config.py` with logical paths, source versions, seeds, and cohort
  parameters;
- `pipeline/io_utils.py` with DuckDB and chunked-read helpers;
- `pipeline/source_inventory.py` CLI for `reports/source_inventory.json`;
- machine-readable inventory of source files, sizes, headers, and checksums;
- synthetic fixtures that mirror only required schemas.

Required tests:

- path resolution;
- source-version validation;
- bounded query behavior;
- missing-file and missing-column errors.

Exit gate:

- no implementation requires loading a full large table into pandas;
- source inventory is reproducible without printing records.

Implemented command:

```powershell
uv run python -m pipeline.source_inventory
```

Latest local metadata-only run:

- sources present: 3;
- files inventoried: 79;
- missing expected files: 0;
- generated artifact: ignored `reports/source_inventory.json`.

## Milestone 2: Cohorts

Status: implemented for broad adult ICU/unit-stay cohorts; sepsis sub-cohort
definition remains pending approval.

### MIMIC-IV

Define adult ICU stays using `icustays`, `patients`, and `admissions`.
Default rule implemented: first ICU stay per admission.

### eICU

Define ICU unit stays from `patient.csv.gz`. Decide and document treatment of
multiple unit stays and age values such as `> 89`.

Default rule implemented: adult unit stays from `patient.csv.gz`; `> 89` ages
are top-coded to age 90 for filtering and flagged as top-coded.

### Sepsis

Approve a source-specific sepsis definition, terminology version, time window,
and comparability strategy before coding.

Deliverables:

- source cohort ID tables;
- attrition manifests;
- source-specific and unified cohort summaries.

Exit gate:

- keys are unique at the declared unit of analysis;
- every stay resolves to a patient;
- no patient crosses data splits;
- source count differences are explained.

Implemented command:

```powershell
uv run python -m pipeline.cohort
```

Latest local aggregate run:

- MIMIC-IV selected stays: 85,242 from 94,458 adult ICU stays after excluding
  9,216 non-first ICU stays within admission;
- eICU selected stays: 200,234 adult unit stays from 200,859 source unit stays,
  with 95 missing or unparseable ages and 7,081 top-coded age stays;
- unified selected stays: 285,476;
- duplicate `stay_uid` count: 0 for each source and unified cohort;
- generated artifacts: ignored `Dataset/processed/cohorts/*.parquet` and
  ignored `reports/cohort_manifest.json`.

## Milestone 3: Schema and Quality Profiling

Status: implemented for default structured source-table profiling; notebook
visualization remains planned.

Profile:

- row and key counts;
- dtypes and parse failures;
- null rates and missingness patterns;
- duplicates and referential integrity;
- plausible ranges;
- units and unit inconsistency;
- timestamp coverage;
- categorical cardinality; and
- source-specific coding conventions.

Deliverables:

- `pipeline/profile_tables.py`;
- `reports/quality_profile.json`;
- `notebooks/01_schema_quality.ipynb` (planned after profile report review);
- aggregate figures under `reports/figures/` (planned after notebook build).

Exit gate:

- every source table used later has an explicit key and quality assessment;
- sensitive rows and note text are absent from reports.

Implemented command:

```powershell
uv run python -m pipeline.profile_tables
```

Latest local aggregate run:

- configured structured tables: 24;
- completed aggregate profiles: 18;
- table-level scan failures recorded: 6;
- failed scans: MIMIC `prescriptions`, `labevents`, `chartevents`,
  `inputevents`; eICU `medication`, `apachePatientResult`;
- completed profiles showed no duplicate-key excess rows and no referential
  orphan rows across configured checks;
- aggregate plausibility checks flagged out-of-bounds values in selected
  weight, vital-sign, and APACHE variables that require review before feature
  engineering;
- generated artifact: ignored `reports/quality_profile.json`.

The scan failures are treated as data-quality/source-integrity blockers for
those tables until the local compressed files and CSV parser settings are
reviewed. No patient rows or note text are written to the report.

Source integrity follow-up:

```powershell
uv run python -m pipeline.source_integrity
```

Latest local integrity run for the six profiling-blocked files:

- checksum status: 6 mismatched against local `SHA256SUMS.txt` manifests;
- gzip integrity status: 6 failed;
- generated artifact: ignored `reports/source_integrity_failed_tables.json`.

Because all six files fail both checksum and gzip validation, they should be
re-transferred or re-downloaded before any downstream extraction or feature
engineering uses them. Parser fallbacks should only be considered after
checksum and gzip checks pass.

## Execution-Plan Milestone 4: EDA and Dataset Understanding

Status: implemented for aggregate report synthesis, stakeholder briefing, and
figure pack. Detailed notebook EDA over cohort-filtered extracts remains
planned after source scan blockers are reviewed.

Deliverables:

- `pipeline/eda_summary.py`;
- ignored `reports/eda_dataset_understanding.json`;
- ignored `reports/eda_dataset_understanding.md`;
- ignored figure pack under `reports/figures/`.

Implemented command:

```powershell
uv run python -m pipeline.eda_summary
```

Latest local aggregate run:

- source groups summarized: 3;
- files summarized: 79;
- broad adult unified cohort: 285,476 stays and 204,234 patients;
- quality profiles summarized: 18 completed out of 24 configured structured
  tables;
- generated figures: cohort selected stays by source, quality profile status,
  largest completed tables, and quality issue categories;
- key stakeholder message: medication and several large event tables require
  scan/parser review before extraction or feature engineering.

This EDA layer uses only aggregate inventory, cohort, and quality-profile
reports. It does not inspect raw rows, note text, or patient-level records.

## Roadmap Milestone 4: Source-Specific Extraction

Status: not started.

MIMIC domains:

- demographics and admissions;
- diagnoses and procedures;
- labs and ICU vitals;
- prescriptions, pharmacy, eMAR, and input events;
- optional discharge and radiology notes.

eICU domains:

- demographics and unit stays;
- diagnoses;
- labs and vitals;
- medication and infusion;
- allergy, APACHE, treatment, and optional notes.

Deliverables:

- `pipeline/mimic_extract.py`;
- `pipeline/eicu_extract.py`;
- bounded, cohort-filtered Parquet outputs;
- extraction manifests.

Exit gate:

- extraction occurs after cohort filtering;
- source fields and original units remain traceable;
- row multiplication and key loss are tested.

## Milestone 5: Harmonization

Status: not started.

Create a source-tagged common schema for:

- patient, encounter, and stay identifiers;
- demographics;
- conditions;
- medication ingredients or classes;
- laboratory and vital concepts;
- allergies and constraints;
- temporal events; and
- optional note references or embeddings.

Do not blindly concatenate sources. Measure:

- condition mapping coverage;
- medication mapping coverage;
- shared lab/vital coverage;
- unit compatibility;
- cohort and missingness differences; and
- concepts retained only for source-specific models.

Deliverables:

- `pipeline/harmonize.py`;
- mapping tables with version and provenance;
- `notebooks/03_harmonization_and_overlap.ipynb`.

Exit gate:

- unmapped concepts are reported, not silently dropped;
- pooled training remains disabled until reviewed coverage thresholds pass.

## Milestone 6: Temporal Features and Labels

Status: not started.

Define:

- index condition and medication decision time;
- feature lookback window;
- label window;
- handling of repeated or simultaneous medications;
- prior medication allowance;
- negative-candidate strategy; and
- censoring and missing-time rules.

Features may include:

- demographics and admission context;
- diagnosis and comorbidity summaries;
- lab and vital summaries, trends, and abnormality flags;
- severity indicators;
- prior interventions;
- allergy or constraint flags; and
- optional pre-index note representations.

Deliverables:

- `pipeline/features.py`;
- `pipeline/build_training_table.py`;
- data dictionary and artifact manifest.

Exit gate:

- temporal cutoff tests pass;
- patient split integrity passes;
- candidates are derived from training data only;
- outcome and target-proxy features are excluded by default.

## Milestone 7: Baselines and Evaluation

Status: not started.

Implement:

- random baseline;
- global and condition popularity baselines;
- linear baseline;
- XGBoost baseline;
- calibration and candidate-coverage analysis.

Report:

- average precision and ROC AUC as secondary row-level metrics;
- precision, recall, hit rate, NDCG, and MRR at K by ranking group;
- per-condition and subgroup results;
- safety-oriented metrics where valid;
- MIMIC validation and eICU external validation;
- confidence intervals or repeated-seed variability where practical.

Exit gate:

- experiments are reproducible from manifests;
- the held-out test set is used only after model choices are frozen;
- claims do not exceed observational-label validity.

## Milestone 8: Graph Suitability and Hybrid Model

Status: not started.

Before building a GNN, quantify:

- node and edge counts;
- degree distributions;
- connected components;
- relation coverage;
- sparsity;
- patient and medication cold-start rates; and
- leakage risk from graph construction.

Then compare:

- tabular/sequence Transformer only;
- GNN only;
- late fusion;
- cross-attention or learned fusion; and
- simple ensemble baselines.

Deliverables:

- `notebooks/04_graph_suitability.ipynb`;
- reviewed graph schema;
- ablation plan;
- hybrid model only after baseline and graph gates pass.

Exit gate:

- graph edges are training-safe and temporally valid;
- hybrid complexity is justified by held-out evidence.

## Milestone 9: Grounded Explanation

Status: not started.

Assemble:

- local feature or token attribution;
- GNN subgraph or path attribution;
- curated knowledge-graph evidence;
- rule and contraindication checks;
- confidence and calibration;
- contradiction and missing-data handling; and
- provenance logs.

The LLM may convert this evidence into readable language but may not add
unsupported reasons.

Evaluate:

- fidelity to model evidence;
- evidence correctness and coverage;
- stability;
- contradiction detection;
- clinician understandability; and
- audit completeness.

Exit gate:

- every explanation item traces to recorded evidence;
- conflicts and uncertainty are visible.

## Milestone 10: Conversational Interface and Human Evaluation

Status: not started.

Build a structured extraction contract for symptoms, diagnoses, history, labs,
prior interventions, current medications, allergies, and constraints. Missing
required fields should trigger clarification rather than a confident ranking.

Human evaluation must involve appropriate clinical expertise and an approved
study protocol before claims about clinical usefulness.

Exit gate:

- the interface clearly states decision-support limitations;
- clinician authority and audit access are preserved;
- no patient-facing autonomous prescribing behavior is introduced.

## Immediate Next Plan

The next implementation task should be source scan-blocker review, notebook
EDA, or Roadmap Milestone 4 extraction only after reviewing
`reports/quality_profile.json` and `reports/eda_dataset_understanding.md`:

1. Review scan failures and decide whether to re-download/repair local source
   files or add reviewed parser fallbacks.
2. Build `notebooks/01_schema_quality.ipynb` and
   `notebooks/02_distributions_correlations.ipynb` from safe aggregate and
   cohort-filtered extracts.
3. Begin Roadmap Milestone 4 source-specific extraction only for tables that
   pass the relevant quality gates.
4. Add extraction tests for cohort filtering, required columns, and row-count
   manifests.
