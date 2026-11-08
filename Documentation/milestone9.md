# Complete CodexPLAN Step 9 Model-Ready Package

## Summary

Current status: **not fully complete, about 75–80% done for the Phase 8 P0 stack**. MIMIC tabular/ranking artifacts, event sequences, train-only candidate catalog, preprocessing, and empirical graph edges exist. Step 9 remains incomplete because `patient_subgraphs` are missing, `cohort_stays` is not model-ready-shaped, Phase 8 provenance stamps drift to v1 downstream, standalone vocab/data-dictionary deliverables are missing, and eICU is currently coverage-only with zero in-catalog positives.

Roadmap Milestone 9 “Grounded Explanation” is separate and remains **not started**; this plan only completes **CodexPLAN Step 9**.

## Key Changes

1. **Lock the Step 9 P0 completion contract**
   - Define completion as: model-ready Phase 8 P0 artifacts are complete for MIMIC development and honest coverage-only eICU handling.
   - Do not add neural training, pooled MIMIC/eICU training, notes, external DDI, ontology KG, or clinical recommendation claims.
   - Treat full KG/DDI/ontology and supervised eICU performance as P1 unless mapping gates produce usable eICU positives.

2. **Fix provenance and cohort schema first**
   - Add shared feature-version inference for downstream builders so Phase 8 P0 training, preprocessing, graph, evaluation, and manifests stamp `temporal-features-v2`.
   - Add a model-ready `cohort_stays` artifact under the selected Phase 8 P0 output root by joining harmonized cohort/demographics with `cohort_decision_times`.
   - Required columns: source/stay/patient IDs, demographics, stay timing, split, `t0`, prediction time, label-window end, eligibility, and version fields.

3. **Implement `patient_subgraphs`**
   - Add a new read-only-input builder, e.g. `pipeline.patient_subgraphs`.
   - Inputs: `patient_condition_medication`, `event_sequences`, and train-fit `graph_edges`.
   - Output a directory artifact, not a single unwieldy nested table:
     - `subgraph_index.parquet`
     - `subgraph_nodes.parquet`
     - `subgraph_edges.parquet`
     - `subgraph_candidates.parquet`
   - One subgraph per `ranking_group_id`/stay-condition group. Include the query condition, all candidate medication nodes, observed pre-decision lab/vital/intervention nodes that connect through train-fit graph edges, and train-fit edges among included nodes.
   - Preserve split/source for loading, but fit no graph statistics from validation/test/eICU rows.

4. **Add vocabularies and data dictionary deliverables**
   - Export local ignored vocabulary tables for condition tokens, candidate medication tokens, event tokens, and graph node IDs.
   - Generate an aggregate-safe `phase8_p0_model_ready_data_dictionary.json` from schemas only: columns, dtypes, key roles, temporal window, branch use, leakage notes, and artifact paths.
   - Public reports must contain no patient IDs, row samples, note text, raw source text, or clinical rows.

5. **Resolve eICU readiness honestly**
   - Run default `rxnorm_or_atc` and sensitivity `atc3_or_rxnorm` candidate-token builds on Phase 8 P0.
   - If eICU gains in-catalog positive ranking groups, mark it externally evaluable and report counts.
   - If eICU remains zero-positive, mark it explicitly as `coverage_only` in manifests/docs and block external performance claims; do not hold MIMIC Step 9 completion hostage to unsupported cross-source labels.

6. **Update execution wrappers and docs**
   - Extend `scripts/calculco/phase8_p0_model_ready.sh` to run: features if needed, training table, preprocessing, graph suitability, patient subgraphs, vocab/data dictionary, and final model-ready manifest.
   - Update `WORKFLOWS.md`, `TESTING.md`, `Documentation/DataFoundationRoadmap.md`, and `CHANGELOG.md` to reflect the completed P0 package and remaining P1 deferrals.

## Task Dependencies

1. Provenance/version inference must land before rebuilding any Phase 8 P0 artifacts.
2. Model-ready `cohort_stays` depends on `cohort_decision_times`.
3. `patient_subgraphs` depends on `patient_condition_medication`, `event_sequences`, and `graph_edges`.
4. Vocabulary/data-dictionary generation depends on all final Step 9 artifacts.
5. eICU readiness review depends on both default and ATC3-sensitivity training-table manifests.
6. Documentation updates depend on final artifact schemas and aggregate manifests.

## Test Plan

- Add synthetic unit tests for:
  - downstream builders inferring `temporal-features-v2`;
  - model-ready `cohort_stays` containing split and prediction-time columns;
  - subgraph construction with train-only graph edges, cold candidates, validation/test/eICU rows, and no future events;
  - vocabulary/data-dictionary generation without row samples or restricted columns;
  - eICU zero-positive behavior producing coverage-only status.
- Run focused checks:
  - `uv run pytest tests/test_features.py tests/test_build_training_table.py tests/test_preprocessing.py tests/test_graph_suitability.py`
  - new tests for `patient_subgraphs` and model-ready dictionary/package generation
  - `uv run ruff check .`
- Protected-data acceptance on Calculco:
  - run the Phase 8 P0 OAR model-ready chain;
  - review aggregate manifests for split integrity, feature version consistency, row counts, eICU positive-group status, graph leakage audit, subgraph counts, and data-safety flags.

## Acceptance Criteria

- Phase 8 P0 has completed artifacts for: `cohort_stays`, `patient_stay_features`, `patient_condition_medication`, `event_sequences`, `graph_edges`, `patient_subgraphs`, vocabularies, split manifest, candidate catalog, preprocessing artifacts, data dictionary, and final model-ready manifest.
- All model-ready artifacts consistently report `temporal-features-v2`.
- MIMIC patient-level split integrity remains zero-overlap.
- Graph edges and subgraphs use train-fit graph statistics only.
- Reports are aggregate-only and disclosure-safe.
- eICU is either externally evaluable with positive groups or explicitly labeled coverage-only with no performance claims.
