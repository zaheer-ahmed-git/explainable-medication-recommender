# Explainable Conversational Medication Recommender

Research foundation for a clinician-facing medication recommendation system
that combines conversational clinical understanding, hybrid
Transformer-GNN ranking, and grounded explanations.

The central research question is:

> How can a medical conversational recommender generate medication rankings
> that are accurate, faithful, grounded, and clinically understandable?

This repository is research software. It is not a medical device, does not
prescribe medication, and must not replace professional clinical judgment.

## Current Status

The repository is currently in the data-foundation and architecture stage.

- The research objective and target architecture are documented.
- Licensed MIMIC-IV v3.1, MIMIC-IV-Note v2.2, and eICU-CRD v2.0 data are
  on Calculco protected storage (`$DATASET_ROOT`); the ignored `Dataset/`
  path is the default when data is colocated with the checkout.
- A previous synthetic preprocessing and ranking prototype is retained in the
  ignored `DepreciatedCode/` directory for reference.
- Python dependencies for analysis, DuckDB processing, baseline modeling, and
  testing are declared in `pyproject.toml`.
- The active `pipeline/` currently includes source-inventory helpers and adult
  ICU/unit-stay cohort materialization for MIMIC-IV and eICU, plus
  aggregate-only source-table quality profiling, EDA briefing synthesis,
  report-gated source extraction CLIs, and Milestone 5 harmonization for
  cohort stays, demographics, conditions, medications, labs, vitals,
  allergies, interventions, and temporal events. Milestone 6 temporal feature,
  split, candidate-catalog, observed-label, and train-fitted preprocessing
  artifact builders are also implemented. Phase 8 P0 feature families are
  available as an isolated `--feature-set phase8_p0` ablation path, pending
  protected-data reruns and promotion review. Milestone 7 baseline evaluation is implemented for aggregate
  coverage, deterministic random, global-popularity, condition-popularity,
  linear, and XGBoost baselines; validation winner frozen as `xgboost` and
  final-mode held-out MIMIC test metrics are recorded in
  `reports/milestone7_baseline_evaluation.json`. Milestone 8 graph-readiness
  tooling is implemented and the protected-data graph gate passed for
  graph-ablation readiness. Milestone 8B graph-aware ablation tooling is
  implemented for graph-only XGBoost, graph-augmented XGBoost, late fusion, and
  simple ensemble comparisons against the frozen XGBoost reference; protected
  Milestone 8B ablation runs are still pending.
- Focused synthetic tests cover the current source-inventory, cohort,
  profiling, EDA-summary, extraction, and Milestone 5 harmonization contracts.
  Additional synthetic tests cover Milestone 6 temporal cutoffs, censoring,
  split integrity, train-only candidates, weak observational labels, and the
  initial Milestone 7 baseline metrics/reporting scaffold, including learned
  linear and XGBoost baselines. Milestone 8 synthetic tests cover train-only
  graph fitting, cold-start reporting, sparse graphs, and report safety.
  Milestone 8B synthetic tests cover graph-feature joins, final-mode gating,
  cold-start flags, fusion, eICU coverage-only behavior, and report safety. The
  full Transformer-GNN model remains planned.

Do not interpret the poster's illustrative medication table or planned system
diagram as a clinically validated implementation.

## Target System

The proposed system has four cooperating layers:

1. **Conversational understanding:** an LLM extracts a structured patient
   profile from clinical dialogue, notes, and structured EHR fields.
2. **Hybrid recommendation:** a Transformer models patient-context and temporal
   interactions while a heterogeneous GNN models patient, diagnosis,
   medication, laboratory, and knowledge relations.
3. **Grounded explainability:** local attribution, graph evidence, clinical
   rules, uncertainty, and provenance are assembled before an LLM verbalizes
   the evidence.
4. **Clinician review:** ranked medication candidates are presented with
   rationale, warnings, uncertainty, and an audit trail.

Recommendation generation and explanation generation remain separate so that
the explanation is tied to actual model evidence instead of a plausible
free-form narrative.

See [ARCHITECTURE.md](ARCHITECTURE.md) for boundaries and
[Documentation/DataFoundationRoadmap.md](Documentation/DataFoundationRoadmap.md)
for the implementation sequence.

## Data Strategy

The project uses:

- **MIMIC-IV v3.1** for deep single-center hospital and ICU development data.
- **MIMIC-IV-Note v2.2** for de-identified discharge and radiology text.
- **eICU-CRD v2.0** for multi-center ICU data and external validation.

The preferred evaluation design is development on MIMIC-IV and external
validation on eICU after careful schema and vocabulary harmonization. Pooled
training is optional and must not be attempted until concept coverage, units,
cohort definitions, and label semantics are demonstrably compatible.

Raw PhysioNet data is licensed and must never be committed or redistributed.

## Planned Data Product

The main modeling artifact is a candidate-ranking table with one row per:

```text
patient/stay + condition + candidate medication
```

Each row should include a prescription label, pre-decision patient context,
source and cohort provenance, a deterministic patient-level split, and explicit
temporal boundaries. Observed prescribing is a historical label, not proof that
a medication is clinically optimal.

## Repository Guide

- `Documentation/ResearchDetail.md`: current research framing and contribution.
- `Documentation/DataFoundationRoadmap.md`: phased implementation plan.
- `Documentation/PosterPresentationGuide.md`: poster explanation and Q&A.
- `Documentation/Milestone6FeatureLabelDictionary.md`: feature and label
  artifact schemas, temporal contract, and observational-label caveats.
- `Documentation/Milestone8.md`: graph-readiness contract, reports, and gates.
- `Documentation/Milestone8B.md`: graph-aware ablation contract, reports, and
  final-mode gate.
- `Documentation/HybridModelFeatureStrategy.md`: planned Transformer/GNN
  feature boundaries and selection gates (post-8B; not neural training yet).
- `visualization/`: aggregate-only Phase 4-9 meeting figure-pack generator.
- `Documentation/SimilarPapers.md`: related-work notes.
- `FinalPosterCDS.pdf`: research poster.
- `ARCHITECTURE.md`: system and data architecture.
- `WORKFLOWS.md`: repeatable development and research workflows.
- `TESTING.md`: verification strategy.
- `CODE_REVIEW.md`: review checklist.
- `AGENTS.md`: durable instructions for coding agents.
- `.agents/skills/`: reusable repository workflows.

Local-only directories:

- `Dataset/`: licensed source data and generated data artifacts.
- `DepreciatedCode/`: synthetic prototype and historical artifacts.

## Environment

Development runs on **ULCO Calculco**. Machine-specific paths are gitignored;
export `PROJECT_HOME`, `DATASET_ROOT`, and `WORK_SCRATCH` before pipeline
commands.

See [Documentation/Environment.md](Documentation/Environment.md) for path
variables, gitignored files, and agent rules. Calculco platform notes:
[Documentation/CalculcoSetup.md](Documentation/CalculcoSetup.md).

This project uses Python 3.13 and `uv` exclusively.

### Path overrides

On Calculco (code in home, data on protected NFS), use **gitignored**
per-machine files — not committed paths:

```bash
cp .env.example .env.calculco
# edit exports, then: set -a && source .env.calculco && set +a
```

Variables: `PROJECT_HOME`, `DATASET_ROOT`, `DATA_PROTECTED`, `REPORTS_ROOT`,
`WORK_SCRATCH`. `pipeline/config.py` reads them when set.

### Setup and verification (lightweight)

Run on the Calculco login node for routine development:

```bash
uv sync
uv run ruff check .
```

```bash
uv run pytest
```

### Pipeline CLIs (protected data required)

The commands below read licensed data via `DATASET_ROOT`. They are bounded but
can be expensive on large tables. Run interactively only for small scopes; submit
full cohort-filtered extraction via OAR (`scripts/calculco/` and
`WORKFLOWS.md`).

Generate the metadata-only source inventory with:

```bash
uv run python -m pipeline.source_inventory
```

Build local ignored adult ICU/unit-stay cohort artifacts with:

```bash
uv run python -m pipeline.cohort
```

Build the aggregate-only source quality profile with:

```bash
uv run python -m pipeline.profile_tables
```

Build the aggregate EDA summary, stakeholder brief, and figure pack with:

```bash
uv run python -m pipeline.eda_summary
```

Build the aggregate Phase 4-9 meeting visualization pack with:

```bash
uv run python -m visualization.phase4_to_9
```

The generated meeting pack is written under ignored `visualization/figures/`
plus `visualization/meeting_figure_pack.md` and
`visualization/meeting_figure_pack.json`. It reads aggregate report manifests
only and does not inspect raw clinical rows.

Check source-file integrity for profiling-blocked files with:

```bash
uv run python -m pipeline.source_integrity
```

Run a manifest-wide integrity audit across MIMIC-IV, MIMIC-IV-Note, and eICU
with:

```bash
uv run python -m pipeline.source_integrity --all-manifest-files
```

Build cohort-filtered local extracts after cohort, quality-profile, and
source-integrity gates are available:

```bash
uv run python -m pipeline.mimic_extract
uv run python -m pipeline.eicu_extract
```

Build the Milestone 5 harmonized cohort-stay, demographics, condition,
medication, lab, vital, allergy, intervention, and temporal-event artifacts
after extracts and local RxNorm/ATC mapping files are available under
`Dataset/mappings/medications/`. Conditions additionally gain optional shared
roll-up tokens (CCSR/CCS/GEM/chapter/structural category, curated eICU text, and
project groups such as sepsis) from optional files under
`Dataset/mappings/conditions/`; missing files degrade gracefully. Fetch the
authoritative CCSR/CCS/GEM reference files (and a derived ICD chapter table)
with `uv run python scripts/fetch_condition_reference_files.py` (needs network),
and inventory diagnosis concepts / emit review-ready templates for the
curation-only files with `uv run python scripts/build_condition_mappings.py`,
then run:

```bash
uv run python -m pipeline.harmonize
```

Large lab/vital harmonization is split into smaller source-query/hash batches
before final combination. On Calculco, tune this with
`--domain-materialization-batches` or the OAR `HARMONIZE_DOMAIN_BATCHES`
environment variable when DuckDB reports out-of-memory errors.

The harmonization reports are aggregate-only:
`reports/harmonization_manifest.json`,
`reports/harmonization_coverage.json`, `reports/unmapped_concepts.json`,
`reports/condition_normalization_coverage.json`, and
`reports/eicu_diagnosis_text_mapping_review.csv`. See
`Documentation/ConditionNormalization.md` for the frozen condition contract.

Build Milestone 6 temporal feature artifacts and the observed-label ranking
table after harmonization coverage and mapping gates are reviewed:

```bash
uv run python -m pipeline.features
uv run python -m pipeline.build_training_table
uv run python -m pipeline.preprocessing
```

These write patient-level local artifacts under
`Dataset/processed/features/` and `Dataset/processed/training/`; aggregate-only
manifests are `reports/milestone6_feature_manifest.json` and
`reports/training_table_manifest.json`, plus
`reports/preprocessing_manifest.json` for train-fitted imputation, scaling,
encoding, and categorical vocabulary metadata. See
`Documentation/Milestone6FeatureLabelDictionary.md`. For large protected-data
runs, `patient_stay_features` and `event_sequences` are stay-hash-batched; tune
with `--stay-feature-batches` / `STAY_FEATURE_BATCHES` and
`--event-sequence-batches` / `EVENT_SEQUENCE_BATCHES`.

To build the optional Phase 8 P0 feature-ablation stack without touching the
default roots:

```bash
uv run python -m pipeline.features --feature-set phase8_p0 \
  --features-root "$DATASET_ROOT/processed/phase8_p0/features" \
  --manifest "$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json"
```

The full isolated sequence is documented in `WORKFLOWS.md`.

Evaluate Milestone 7 coverage and transparent baselines after the Milestone 6
manifests are reviewed:

```bash
uv run python -m pipeline.evaluate_baselines
uv run python -m pipeline.evaluate_baselines --baseline linear --baseline xgboost
```

Development mode writes aggregate-only reports
`reports/milestone7_coverage_report.json` and
`reports/milestone7_baseline_evaluation.json`, with local row-level scores and
learned model artifacts under ignored
`Dataset/processed/evaluation/milestone7/`. Held-out test metrics are blocked unless final mode is explicit:

```bash
scripts/calculco/submit_evaluate_baselines.sh final
```

Build Milestone 8 graph-readiness artifacts after Milestone 6 artifacts and
Milestone 7 frozen selection are available:

```bash
uv run python -m pipeline.graph_suitability
```

This writes concept-level local graph artifacts under ignored
`Dataset/processed/graph/milestone8/` and aggregate-only reports
`reports/milestone8_graph_schema.json`,
`reports/milestone8_graph_suitability.json`, and
`reports/milestone8_ablation_plan.json`. It does not train the Transformer-GNN
model or make clinical recommendation claims.

Run Milestone 8B graph-aware ablations only after Milestone 7 final evaluation
and the Milestone 8 graph gate have passed:

```bash
uv run python -m pipeline.graph_ablation
```

For protected-data scale, submit the development run through OAR, then run final
mode only after `reports/milestone8b_frozen_selection.json` exists:

```bash
scripts/calculco/submit_graph_ablation.sh development
scripts/calculco/submit_graph_ablation.sh final
```

Milestone 8B writes local row-level scores and model artifacts under ignored
`Dataset/processed/evaluation/milestone8b/` and aggregate-only reports
`reports/milestone8b_graph_feature_manifest.json`,
`reports/milestone8b_ablation_evaluation.json`, and
`reports/milestone8b_frozen_selection.json`. It is a graph-aware ablation gate,
not validated clinical recommendation behavior or full Transformer-GNN training.

Do not use `pip`, Poetry, Conda, global Python, or system site-packages.

## Working Principles

- Plan before risky or multi-file changes.
- Use bounded DuckDB queries or chunked readers for large compressed tables.
- Split by patient and apply temporal cutoffs before modeling.
- Keep source-specific extraction separate from harmonization.
- Log cohort, feature, label, split, and model provenance.
- Use synthetic fixtures in tests.
- Review the diff, verification results, data-safety implications, and research
  claims before merging.

## Research References

The project direction and related-work notes are maintained in
`Documentation/ResearchDetail.md` and `Documentation/SimilarPapers.md`.
Dataset users must also cite the official MIMIC-IV, MIMIC-IV-Note, and eICU-CRD
publications and comply with their PhysioNet data-use agreements.

## Security and Contributions

Read [SECURITY.md](SECURITY.md) before handling clinical data or reporting a
vulnerability. Contribution expectations are in
[CONTRIBUTING.md](CONTRIBUTING.md).

No open-source license has been selected for this repository. Unless a license
is added by the maintainers, no permission to reuse or redistribute the code is
granted.
