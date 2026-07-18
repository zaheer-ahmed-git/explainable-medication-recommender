# Data Foundation Roadmap

## Purpose

This roadmap turns the research architecture into a sequence of verifiable
data and modeling milestones. It supersedes status assumptions in older plans
when they conflict with the current working tree.

Last reviewed: 2026-07-11.

## Current Baseline

Available locally under the ignored `Dataset/` directory:

- MIMIC-IV v3.1 hospital and ICU tables;
- MIMIC-IV-Note v2.2 discharge and radiology notes; and
- eICU-CRD v2.0 tables.

The source files include multi-gigabyte compressed CSV tables. They require
DuckDB projection/filtering or chunked streaming.

The active repository contains source-inventory, adult ICU/unit-stay cohort
materialization, aggregate source-table quality profiling, aggregate EDA
briefing synthesis, report-gated source extraction CLIs, full local
cohort-filtered MIMIC/eICU extraction runs, and Milestone 5 harmonization for
cohort stays, demographics, conditions, medications, labs, vitals, allergies,
interventions, and temporal events with synthetic tests. Milestone 6 temporal
feature construction, patient splitting, train-only candidate catalogs, and
observed-label ranking-table builders are implemented with synthetic tests and
protected-data materialization. Milestone 7 baseline evaluation is implemented
through learned linear and XGBoost baselines with frozen validation selection.
Milestone 8 graph-readiness tooling is implemented for train-only concept-level
graph artifacts and aggregate suitability reports; protected-data graph
materialization and graph neural models remain pending. The ignored
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
  extract_utils.py
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

```bash
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

Approved 2026-07-04: coded sepsis definition (A1) now with Sepsis-3 (A2)
deferred until the `chartevents`/`inputevents` refresh, and an index-condition
policy of B1 (all CCS/CCSR categories) for the first Milestone 6 run then B3
(sepsis project-group deep dive) for Milestone 7. See
`Documentation/SepsisCohortAndIndexConditionPolicy.md` for the code set and
implementation steps.

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

```bash
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

```bash
uv run python -m pipeline.profile_tables
```

Latest local aggregate run (2026-06-18):

- configured structured tables: 24;
- completed aggregate profiles: 22;
- table-level scan failures recorded: 2;
- failed scans: MIMIC `chartevents` and `inputevents` (stale relative to the
  corrected local files verified on 2026-06-30);
- completed profiles showed no duplicate-key excess rows and no referential
  orphan rows across configured checks;
- aggregate plausibility checks flagged out-of-bounds values in selected
  weight, vital-sign, and APACHE variables that require review before feature
  engineering;
- generated artifact: ignored `reports/quality_profile.json`.

Re-run `pipeline.profile_tables` after source-file correction so extraction
gates and EDA summaries reflect the current `chartevents` and `inputevents`
files. No patient rows or note text are written to the report.

Source integrity follow-up:

```bash
uv run python -m pipeline.source_integrity
uv run python -m pipeline.source_integrity --all-manifest-files
```

Latest targeted integrity audit of the six previously profiling-blocked files
(2026-06-30):

- checksum/gzip passed: MIMIC `prescriptions`, MIMIC `labevents`, MIMIC
  `chartevents`, MIMIC `inputevents`, eICU `medication`, and eICU
  `apachePatientResult`;
- generated artifact: ignored `reports/source_integrity_failed_tables.json`.

Earlier full local integrity audit across all files listed in configured
`SHA256SUMS.txt` manifests (2026-06-18):

- files checked: 70;
- checksum matches: 66;
- checksum mismatches with gzip failures: MIMIC `icu/chartevents.csv.gz` and
  MIMIC `icu/inputevents.csv.gz` (superseded by the 2026-06-30 targeted
  re-audit on corrected local files);
- MIMIC-IV-Note v2.2: 5 manifest entries checked; 3 checksum matches; 2
  manifest/layout reconciliations where `SHA256SUMS.txt` lists
  `note/discharge_detail.csv.gz` and `note/radiology_detail.csv.gz`, while the
  configured and locally present source files are the original uncompressed
  `note/discharge_detail.csv` and `note/radiology_detail.csv`;
- generated artifact: ignored `reports/source_integrity_all_manifest_files.json`.

Any mismatched, truly missing, or gzip-failing file should be re-transferred,
re-downloaded, or reconciled against the official source package before
downstream extraction or feature engineering uses it. Configured uncompressed
MIMIC-IV-Note detail files are not treated as corrupt merely because a manifest
entry uses a `.csv.gz` suffix. Parser fallbacks should only be considered after
checksum and gzip checks pass.

## Execution-Plan Milestone 4: EDA and Dataset Understanding

Status: implemented for aggregate report synthesis, stakeholder briefing, and
figure pack. The latest EDA summary predates corrected `chartevents` and
`inputevents` files and full extraction runs; re-run after refreshed quality
profiles. Detailed notebook EDA over cohort-filtered extracts remains planned.

Deliverables:

- `pipeline/eda_summary.py`;
- ignored `reports/eda_dataset_understanding.json`;
- ignored `reports/eda_dataset_understanding.md`;
- ignored figure pack under `reports/figures/`.

Implemented command:

```bash
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
- key stakeholder message at run time: medication and several large event tables
  required scan/parser review before extraction or feature engineering (MIMIC
  `chartevents` and `inputevents` source files now pass integrity; refresh
  quality profiles and re-run EDA to update this message).

This EDA layer uses only aggregate inventory, cohort, and quality-profile
reports. It does not inspect raw rows, note text, or patient-level records.

## Roadmap Milestone 4: Source-Specific Extraction

Status: implemented for report-gated extraction CLIs with synthetic contract
tests. Full local cohort-filtered extraction runs completed on 2026-06-28;
aggregate manifests reviewed.

MIMIC domains:

- demographics and admissions, extracted;
- diagnoses and procedures, extracted;
- labs, extracted;
- ICU procedure events, extracted;
- ICU input events, implemented as a gated spec; skipped on the 2026-06-28 run
  because the 2026-06-18 quality profile still recorded `scan_failed` (local
  `inputevents.csv.gz` now passes the 2026-06-30 integrity audit; re-profile and
  re-extract to materialize);
- prescriptions, extracted;
- charted ICU vitals (`chartevents`), implemented as a gated spec restricted to
  curated core-vital itemids (`MIMIC_CHARTEVENTS_VITAL_ITEMIDS`); gated on a
  refreshed quality/integrity profile like `inputevents`, so it materializes
  after re-profiling;
- pharmacy, eMAR, and POE extraction specs not yet added to the CLI;
- optional discharge and radiology notes remain deferred.

eICU domains:

- demographics and unit stays, extracted;
- diagnoses, labs, vitals, medication, infusion, allergy, APACHE, and treatment,
  extracted;
- optional notes remain deferred.

Deliverables:

- `pipeline/mimic_extract.py`;
- `pipeline/eicu_extract.py`;
- bounded, cohort-filtered Parquet outputs;
- extraction manifests.

Implemented commands:

```bash
uv run python -m pipeline.mimic_extract
uv run python -m pipeline.eicu_extract
```

Latest local cohort-filtered extraction runs (2026-06-28):

- MIMIC: 10 of 11 configured tables completed; 1 skipped (`mimic_inputevents`,
  stale quality gate); ~50.1M cohort-filtered rows across completed tables;
  generated artifact: ignored `reports/mimic_extraction_manifest.json` and
  ignored Parquet under `Dataset/processed/extracts/mimiciv/`.
- eICU: 12 of 12 configured tables completed; ~230.1M cohort-filtered rows;
  generated artifact: ignored `reports/eicu_extraction_manifest.json` and ignored
  Parquet under `Dataset/processed/extracts/eicu/`.

Current implementation notes:

- `pipeline/extract_utils.py` centralizes required-column checks, cohort joins,
  source quality/integrity gates, local Parquet writes, and aggregate manifests.
- Extraction manifests are aggregate-only; patient-level extracted rows are
  written only under ignored `Dataset/processed/extracts/`.
- Synthetic tests verify cohort filtering before materialization, blocked-table
  skipping, and aggregate-only manifests.
- MIMIC `inputevents` re-extraction and new `chartevents` extraction require a
  refreshed `reports/quality_profile.json` after the corrected source files are
  profiled.

Exit gate:

- extraction occurs after cohort filtering;
- source fields and original units remain traceable;
- row multiplication and key loss are tested.

## Milestone 5: Harmonization

Status: implemented for the CLI, mapping-resource gate, cohort-stay,
demographics, semantically normalized conditions, RxNorm/ATC-mapped medication,
lab, vital, allergy, intervention, and temporal-event artifacts, plus aggregate
coverage/unmapped reports and the aggregate overlap notebook. Conditions now add
optional shared roll-up tokens (CCSR/CCS/GEM/chapter/structural category, eICU
curated text, and project groups such as sepsis) while preserving source-native
codes; missing condition mapping files degrade gracefully rather than failing.
Reviewed coverage thresholds remain a gate before any pooled MIMIC/eICU
training. Shared condition vocabulary and roll-up level are resolved for this
stage in `Documentation/ConditionNormalization.md`.
Harmonization also filters eICU medication rows marked `drugordercancelled`,
deduplicates repeated domain events, and records aggregate cleanup and
source/stay join-integrity counts.

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

Implemented command:

```bash
uv run python -m pipeline.harmonize
```

Latest local aggregate run (2026-07-01):

- harmonization status: completed;
- harmonized artifacts: 9 of 9 configured tables (`cohort_stays`,
  `demographics`, `conditions`, `medications`, `labs`, `vitals`, `allergies`,
  `interventions`, `temporal_events`);
- unified cohort stays harmonized: 285,476;
- generated artifacts: ignored `Dataset/processed/harmonized/*.parquet`,
  `reports/harmonization_manifest.json`, `reports/harmonization_coverage.json`,
  and `reports/unmapped_concepts.json`.

Current implementation notes:

- Medication harmonization requires local ignored mapping files under
  `Dataset/mappings/medications/`:
  `mimic_ndc_rxnorm_atc.csv` and `eicu_drug_rxnorm_atc.csv`.
- Condition normalization uses optional local ignored files under
  `Dataset/mappings/conditions/` (CCSR/CCS/GEM/chapter/eICU-text/project-group).
  Missing files degrade to structural ICD categories and source-native tokens;
  `scripts/build_condition_mappings.py` writes review-ready templates. See
  `Documentation/ConditionNormalization.md`.
- Missing or malformed mapping resources produce
  `reports/harmonization_manifest.json` with
  `failed_missing_mapping_resources` and a nonzero CLI exit.
- Aggregate coverage and unmapped reports are written to
  `reports/harmonization_coverage.json` and
  `reports/unmapped_concepts.json` when harmonization runs.
- Aggregate cleanup counts include cancelled eICU medication orders,
  event-level deduplication summaries, and join-integrity checks against the
  harmonized cohort. Join-integrity failures set the manifest status to
  `failed_join_integrity`.
- Large lab and vital harmonization uses split source-query/hash-batched
  materialization (`--domain-materialization-batches`) before combining the
  canonical `labs.parquet` and `vitals.parquet` files, reducing peak DuckDB
  memory during OAR runs.
- Harmonized artifacts are written under `Dataset/processed/harmonized/`:
  `cohort_stays.parquet`, `demographics.parquet`, `conditions.parquet`,
  `medications.parquet`, `labs.parquet`, `vitals.parquet`,
  `allergies.parquet`, `interventions.parquet`, and
  `temporal_events.parquet`.
- Lab and vital concepts use reviewed mappings only when present; otherwise
  source-native tokens and units are preserved and reported with aggregate unit
  availability/compatibility counts.

Exit gate:

- unmapped concepts are reported, not silently dropped;
- every harmonized artifact records source, cohort version, extraction
  version, mapping version, harmonization version, and generation timestamp;
- pooled training remains disabled until reviewed coverage thresholds pass.

## Milestone 6: Temporal Features and Labels

Status: **completed** for code, synthetic tests, and protected-data
materialization (OAR jobs 830 and 1055 on ritchie/chimay, 2026-07-05/06). See
`Documentation/Milestone6MaterializationReview.md` and
`reports/milestone6_materialization_review.json`. Review catalog coverage and
out-of-catalog positives before Milestone 7 baselines.

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
- `pipeline/preprocessing.py`;
- `Documentation/Milestone6FeatureLabelDictionary.md`;
- ignored feature artifacts under `Dataset/processed/features/`;
- ignored training artifacts under `Dataset/processed/training/`;
- aggregate-only manifests:
  `reports/milestone6_feature_manifest.json`,
  `reports/training_table_manifest.json`, and
  `reports/preprocessing_manifest.json`.

Implemented commands:

```bash
uv run python -m pipeline.features
uv run python -m pipeline.build_training_table
uv run python -m pipeline.preprocessing
```

Current implementation notes:

- `pipeline.features` writes `cohort_decision_times.parquet`,
  `patient_stay_features.parquet`, and `event_sequences.parquet`.
- `pipeline.build_training_table` writes `split_manifest.parquet`,
  `candidate_catalog.parquet`, and `patient_condition_medication.parquet`.
- `pipeline.preprocessing` fits imputation, scaling, encoding, and categorical
  vocabularies on MIMIC train rows only, then saves local ignored preprocessing
  artifacts and an aggregate manifest.
- Default temporal contract: `t_pred = t0 + 24h`; label window is medication
  starts in `(24h, 48h]`.
- MIMIC uses deterministic patient-level train/validation/test splits from the
  configured seed; eICU is assigned to `external`.
- Candidate catalogs are condition-specific and learned from MIMIC train
  positives only.
- Untimed condition rows define ranking groups but are excluded from default
  event-sequence features.
- Pre-decision medication events are excluded from default event sequences to
  reduce target-proxy leakage risk; a CLI flag allows reviewed experiments to
  include them.
- `patient_stay_features` is materialized with configurable stay-hash batches
  to bound DuckDB memory on large lab/vital aggregates.
- `event_sequences` is materialized with a staged pre-decision event file and
  configurable stay-hash batches before final single-file combination to bound
  DuckDB window-function memory on large `temporal_events` inputs.
- Phase 8 P0 feature engineering is implemented as an isolated optional
  `pipeline.features --feature-set phase8_p0` path that writes
  `temporal-features-v2` with train-fit condition presence columns, lab/vital
  trend summaries, explicit missingness indicators, and aggregate-only OOV
  counts. Default roots and `temporal-features-v1` remain unchanged until the
  protected-data promotion gate passes.
- Reports are aggregate-only; patient-level feature/training/preprocessing
  artifacts remain local and ignored.
- Protected-data materialization runs via OAR wrappers
  `scripts/calculco/features.sh`, `scripts/calculco/build_training_table.sh`,
  and the `scripts/calculco/milestone6.sh` chain. Phase 8 P0 isolated feature
  and model-ready builds use `scripts/calculco/phase8_p0_features.sh` and
  `scripts/calculco/phase8_p0_model_ready.sh`; re-profiling before an
  `inputevents` re-extract uses `scripts/calculco/profile_tables.sh`.

Exit gate:

- temporal cutoff tests pass;
- patient split integrity passes;
- candidates are derived from training data only;
- outcome and target-proxy features are excluded by default.

## Milestone 7: Baselines and Evaluation

Status: core baseline evaluation complete on protected data. Development
selection completed on Calculco job 2084; validation winner `xgboost` is frozen
in `reports/milestone7_frozen_selection.json`, and final-mode held-out MIMIC
test metrics are recorded in `reports/milestone7_baseline_evaluation.json`.
Sepsis/B3 headline reporting remains pending.

Implement:

- random baseline, implemented in `pipeline.evaluate_baselines`;
- global and condition popularity baselines, implemented with MIMIC train-only
  fitting in `pipeline.evaluate_baselines`;
- linear baseline, implemented in `pipeline.learned_baselines`;
- XGBoost baseline, implemented in `pipeline.learned_baselines`;
- calibration and candidate-coverage analysis, implemented for non-learned and
  learned baselines.

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

## Milestone 8: Graph Suitability

Status: complete for the graph-readiness gate. The code, synthetic tests,
aggregate report schemas, OAR wrapper, aggregate-only review notebook, and
protected-data materialization are available. The protected-data graph gate
passed for graph ablation readiness; hybrid GNN/Transformer training remains
separate from this milestone.

Before building a GNN, `pipeline.graph_suitability` quantifies:

- node and edge counts;
- degree distributions;
- connected components;
- relation coverage;
- sparsity;
- patient and medication cold-start rates; and
- leakage risk from graph construction.

Implemented command:

```bash
uv run python -m pipeline.graph_suitability
```

Current implementation notes:

- graph edges are concept-level local artifacts under
  `Dataset/processed/graph/milestone8/`;
- learned graph statistics are fit from MIMIC train rows only;
- validation/test/eICU rows are used only for aggregate coverage and cold-start
  reporting;
- reports are aggregate-only:
  `reports/milestone8_graph_schema.json`,
  `reports/milestone8_graph_suitability.json`, and
  `reports/milestone8_ablation_plan.json`;
- external DDI, ontology, note, and rule edges remain deferred until
  curated sources and leakage policies are reviewed.

After the graph gate passes, compare in Milestone 8B:

- frozen XGBoost reference;
- graph-only relation-feature XGBoost;
- XGBoost augmented with graph-derived features;
- validation-weighted late fusion; and
- simple ensemble baselines.

Deliverables:

- `pipeline/graph_suitability.py`;
- `scripts/calculco/graph_suitability.sh`;
- `notebooks/04_graph_suitability.ipynb`;
- `Documentation/Milestone8.md`;
- reviewed graph schema and ablation-plan reports;
- hybrid model only after baseline, graph, and ablation gates pass.

Exit gate:

- graph edges are training-safe and temporally valid;
- reports contain no patient-level rows or raw clinical examples;
- hybrid complexity is justified by held-out evidence in a later modeling
  milestone.

## Milestone 8B: Graph-Aware Ablation Gate

Status: implemented for code and synthetic tests; protected-data ablation runs
are pending. This milestone uses the passed Milestone 8 graph gate and the
frozen Milestone 7 XGBoost reference to evaluate whether graph-derived features
or fusion provide enough held-out validation lift to justify deeper neural
GNN/Transformer work.

Implemented command:

```bash
uv run python -m pipeline.graph_ablation
```

Current implementation notes:

- local graph-feature, score, and model artifacts are written under ignored
  `Dataset/processed/evaluation/milestone8b/`;
- aggregate-only reports are
  `reports/milestone8b_graph_feature_manifest.json`,
  `reports/milestone8b_ablation_evaluation.json`, and
  `reports/milestone8b_frozen_selection.json`;
- graph-derived statistics come only from train-fit Milestone 8 graph edges;
- model selection uses MIMIC validation only, with MIMIC test blocked until
  final mode and a frozen 8B selection are explicit;
- eICU remains coverage-only until in-catalog positive groups exist;
- no new neural framework, external DDI, ontology, note, or rule
  edges are introduced in this milestone.

Exit gate:

- graph-aware candidates must clear the validation lift gate over the frozen
  XGBoost reference before deeper hybrid work is justified;
- final/test evaluation must use the frozen 8B selection;
- no clinical recommendation or full Transformer-GNN claim is made from this
  ablation output alone.

## Pre-Hybrid Feature Strategy (Planning)

Status: documented. Phase 8 P0 structured feature families are implemented as
an isolated ablation path, but not promoted to canonical defaults.

After Milestone 8B, use `Documentation/HybridModelFeatureStrategy.md` as the
canonical planning reference for:

- Transformer versus GNN branch boundaries;
- implemented versus planned feature families aligned with Milestone 6 and 8;
- validation NDCG@10 and Milestone 8B lift gates for feature ablation;
- Phase 2+ deferrals (notes, external DDI/ontology).

This section does not add a new milestone number or authorize neural training
before the 8B exit gate is reviewed.

## Phase 8 P0 Feature Ablation and CodexPLAN Step 9 Rebuild

Status: protected-data package materialized (2026-07-17 model-ready manifest);
Milestone 7/8/8B Phase 8 P0 reports exist; feature promotion gate is
`reject_inconclusive`. CodexPLAN Step 10 graph/hybrid readiness review
(2026-07-18) confirms structure pass and hybrid-lift fail: retain frozen
XGBoost; do not start neural Transformer-GNN yet. See
`Documentation/CodexPLANStep10GraphHybridReadiness.md`.

Execution order:

1. Build isolated `phase8_p0` features with `temporal-features-v2`.
2. Build RxNorm-first and ATC-3-first ranking manifests, train-only
   preprocessing, and a model-ready `cohort_stays` artifact with split and
   decision-window fields.
3. Build train-fit graph edges and normalized `patient_subgraphs`, then local
   condition/medication/event/graph vocabularies.
4. Write the schema-only model-ready data dictionary and aggregate completion
   manifest; mark eICU externally evaluable only when a completed mapping
   strategy has positive ranking groups.
5. Rerun Roadmap Milestone 7 and Milestone 8B development evaluation on the
   isolated roots.
6. Write `reports/phase8_p0_feature_gate_review.json` with
   `pipeline.feature_gate_review`.
7. Promote only if the reviewed validation NDCG@10 lift and secondary-metric
   guardrails pass; otherwise keep current canonical roots and reports.

CodexPLAN Step 9 here means the model-ready artifact rebuild. It is unrelated
to Roadmap Milestone 9 grounded explanation.

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

Milestone 6–8B and CodexPLAN Steps 9–10 are available on the Phase 8 P0 stack.
Graph structure readiness passed; hybrid complexity over transparent baselines
did not. The next tasks are:

1. Keep frozen tabular XGBoost as the development reference; do not start
   neural Transformer-GNN training until a reviewed ablation clears the
   +0.005 NDCG@10 lift gate (see
   `Documentation/CodexPLANStep10GraphHybridReadiness.md`).
2. Prioritize cheaper lifts: medication-mapping coverage (including the ATC-3
   eICU-evaluable path), condition-feature redesign after
   `reject_inconclusive`, and targeted feature ablations under the same metric
   gate.
3. Keep pooled MIMIC/eICU training, full Transformer-GNN claims, and clinical
   recommendation claims disabled until reviewed evidence justifies them.
