# Testing

## Current State

The active working tree contains the source-inventory, adult cohort, aggregate
table-profiling, EDA-summary, source-extraction, and Milestone 5 harmonization
pipeline with focused synthetic tests. Milestone 6 feature and observed-label
builders also have focused synthetic tests. The first Milestone 7
baseline-evaluation scaffold has synthetic tests for coverage, non-learned and
learned baselines, ranking metrics, row-level null metrics, Milestone 6
integration, and final-mode gating. Milestone 8 graph-readiness tooling has
synthetic tests for train-only graph fitting, cold-start reporting, sparse
graphs, and report safety. Milestone 8B graph-aware ablation tooling has
synthetic tests for train-fit graph features, cold-start flags, fusion,
final-mode gating, eICU coverage-only behavior, and report safety. Graph neural
models are still planned and still need their own tests as they are added.
Condition mapping tests also cover active A1/B3 sepsis mapping generation and
`icd_prefix` project-group matching. Training-table tests cover the default
RxNorm-first candidate token strategy and the ATC-3-first coverage-sensitivity
strategy. Preprocessing tests cover train-only imputation/scaling/encoding
artifact fitting and aggregate-only preprocessing manifests.
Phase 8 P0 tests cover train-only condition vocabulary fitting, lab/vital trend
boundaries, explicit missingness columns, aggregate-only feature manifests,
downstream learned-baseline/graph-ablation feature pickup, and the promotion
gate review writer. CodexPLAN Step 9 package tests cover input-derived version
stamps, model-ready cohort timing fields, train-fit patient subgraphs, cold
candidates, future-event exclusion, bounded source-stay hash materialization,
integer-encoded relation-specific edge joins, independent join sharding,
temporary-part cleanup, normalized vocabulary outputs, schema-only data
dictionaries, and eICU coverage-only claim blocking.
Gate-first Stage 1 tests cover immutable contract metadata, pinned versions,
unsafe-column rejection, temporal and medication leakage, patient-fold
isolation, positive-group negative sampling, authoritative ranking-metric
parity, OOF fusion, changed-lock detection, and final-mode blocking. The
protected recovery run remains an HPC verification step. Neural loader,
sequence, loss, and graph-tensor tests remain conditional because Stage 2 is
not implemented.
The Phase 4-9 visualization generator has a focused synthetic test that verifies
aggregate-only meeting-pack generation without raw clinical rows.

Every new active module should arrive with focused tests.

**Execution context:** ULCO Calculco. Export `PROJECT_HOME` and `DATASET_ROOT`
before pipeline CLIs. See `Documentation/Environment.md`.

## Verification Tiers

| Tier | Where | When |
|------|-------|------|
| Lightweight | Calculco login node | Config, unit tests, lint, docs |
| Bounded CLI | Login node (with `DATASET_ROOT`) | Small inventory or manifest checks |
| HPC | OAR compute nodes | Cohort-filtered extraction, large scans |

Do not submit OAR jobs or long-running full-dataset work unless explicitly
requested. See `WORKFLOWS.md` and `scripts/calculco/`.

## Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

During development, prefer a focused path:

```bash
uv run pytest tests/test_config.py tests/test_io_utils.py
uv run pytest tests/test_cohort.py
uv run pytest tests/test_profile_tables.py
uv run pytest tests/test_eda_summary.py
uv run pytest tests/test_source_integrity.py
uv run pytest tests/test_extraction_harmonize.py
uv run pytest tests/test_config.py tests/test_features.py tests/test_build_training_table.py
uv run pytest tests/test_preprocessing.py
uv run pytest tests/test_config.py tests/test_milestone7_baselines.py
uv run pytest tests/test_config.py tests/test_graph_suitability.py
uv run pytest tests/test_patient_subgraphs.py tests/test_model_ready_package.py
uv run pytest tests/test_config.py tests/test_graph_ablation.py
uv run pytest tests/test_feature_gate_review.py tests/test_features.py \
  tests/test_milestone7_baselines.py tests/test_graph_ablation.py
uv run pytest tests/test_training_contract.py tests/test_gate_recovery.py
uv run pytest tests/test_phase4_to_9_visualization.py
uv run pytest tests/test_condition_normalization.py
uv run pytest tests/test_condition_mapping_builder.py
uv run pytest tests/test_cohort.py -k patient_split
```

## Fixture Policy

- Never use raw MIMIC-IV, MIMIC-IV-Note, or eICU rows in tests.
- Build small synthetic tables with invented identifiers and values.
- Include edge cases deliberately: missing identifiers, repeated stays, null
  units, simultaneous events, and boundary timestamps.
- Keep fixtures readable enough that expected behavior can be reviewed.

## Test Layers

### Unit Tests

Cover normalization, schema validation, deterministic splitting, temporal
windows, candidate generation, metrics, and mapping utilities.

### Contract Tests

Verify required input columns, output schemas, identifier uniqueness, allowed
nulls, units, provenance fields, and manifest structure.

### Integration Tests

Run source-specific extraction and harmonization against tiny synthetic CSV or
Parquet fixtures. Confirm filters are applied before materialization.

### Model Tests

Use small deterministic data to verify feature selection, leakage exclusions,
fit/predict shape, ranking groups, metric calculations, and artifact metadata.
Do not assert that a tiny model reaches clinically meaningful performance.

### Documentation Tests

Check that commands, paths, links, and status statements match the repository.

## Required Invariants

- One patient maps to exactly one split.
- Candidate catalogs are learned from training data only.
- Feature timestamps do not exceed the decision cutoff.
- Future outcomes are excluded from predictors by default.
- Source-qualified identifiers remain unique.
- Harmonization reports unmapped concepts instead of silently dropping them.
- Condition normalization preserves source-native tokens, adds shared roll-ups
  only from authoritative fixtures, degrades gracefully when mapping files are
  missing, and drops no diagnosis rows.
- Every artifact records cohort and configuration provenance.

## Full Verification

Before a milestone is considered complete:

1. Run targeted tests.
2. Run the complete test suite.
3. Run Ruff check and format check.
4. Inspect generated manifests and aggregate counts.
5. Review the Git diff.
6. Report checks and any unavailable validation.

Milestone 5 harmonization completion additionally requires the manifest to list
`cohort_stays`, `demographics`, `conditions`, `medications`, `labs`, `vitals`,
`allergies`, `interventions`, and `temporal_events`, with aggregate coverage
and unmapped reports.

Milestone 6 completion additionally requires temporal cutoff tests, censoring
tests, patient split-integrity tests, train-only candidate-catalog tests,
out-of-catalog positive reporting, join-integrity gates, train-only
preprocessing artifact fitting, and aggregate-only manifest checks.

Milestone 7 baseline completion additionally requires coverage and evaluability
review on protected data, learned-baseline manifest metadata when linear or
XGBoost baselines are selected, train-only popularity tests, deterministic
random-score tests, ranking metric tests, report-safety checks, explicit
final/test gating, and Calculco submission via
`scripts/calculco/submit_evaluate_baselines.sh`.

Milestone 8 graph-readiness completion additionally requires train-only graph
fitting tests, no future-event graph edges, no validation/test/eICU graph
fitting, cold-start reporting, sparse-graph handling, aggregate-only report
checks, and Calculco submission via
`scripts/calculco/graph_suitability.sh` when full artifacts are needed.

Milestone 8B graph-ablation completion additionally requires train-fit graph
feature tests, validation-only fusion/selection tests, final/test blocking until
the frozen 8B selection exists, eICU coverage-only checks when positives are
absent, aggregate-only report checks, and Calculco submission via
`scripts/calculco/submit_graph_ablation.sh` when protected-data ablation metrics
are needed.

Phase 8 P0 completion additionally requires synthetic tests for train-only
condition vocabularies, temporal trend cutoffs, report safety, downstream model
feature selection, model-ready cohort timing, version consistency, train-fit
subgraphs, vocabulary/data-dictionary safety, eICU evaluability status, and
gate-review behavior. Protected-data completion requires the successful
`scripts/calculco/phase8_p0_model_ready.sh` OAR chain, an aggregate final
package manifest with all required artifacts, and no promotion until
`reports/phase8_p0_feature_gate_review.json` passes.

Gate-recovery Stage 1 additionally requires contract-lock drift checks,
declared-versus-actual row counts, safe model projections, aggregate-only
reports, train-only patient-fold selection, group-preserving sampling, metric
parity, OOF-only fusion selection, one-shot validation, and final/test
blocking. Protected completion requires the CPU OAR development run and a
reviewed `reports/phase8_p0_gate_recovery_selection.json`; it does not imply
that the gate passed or authorize Stage 2.
