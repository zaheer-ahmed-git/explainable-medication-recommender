# Milestone 8B Execution Plan: Graph-Aware Ablation Gate

## Summary

Milestone 8B evaluates whether the passed Milestone 8 graph is useful for
medication ranking before any neural Transformer/GNN implementation is added.
It compares graph-aware ablations against the frozen Milestone 7 XGBoost
reference using the same aggregate metric machinery and leakage boundaries.

This milestone does not add external DDI, ontology, note, or rule
edges. It does not claim clinical recommendation validity or full hybrid-model
completion.

## Implementation Status

Implemented for code and synthetic tests:

- `pipeline.graph_ablation` builds graph-derived candidate features, trains
  graph-only and graph-augmented XGBoost ablations, computes late-fusion and
  simple-ensemble scores, and writes aggregate-only reports.
- Local row-level features, scores, models, and fusion weights are written under
  `$DATASET_ROOT/processed/evaluation/milestone8b/`.
- Aggregate reports are written to
  `reports/milestone8b_graph_feature_manifest.json`,
  `reports/milestone8b_ablation_evaluation.json`, and
  `reports/milestone8b_frozen_selection.json`.
- `scripts/calculco/graph_ablation.sh` and
  `scripts/calculco/submit_graph_ablation.sh` run protected-data jobs through
  OAR with CPU-only scheduling and bounded DuckDB settings.
- `tests/test_graph_ablation.py` covers train-only graph features, cold-start
  flags, final-mode gating, fusion, eICU coverage-only behavior, and report
  safety.

Protected-data Milestone 8B development and final runs are pending.

## Contract

Inputs:

- Milestone 6 `patient_stay_features.parquet`,
  `candidate_catalog.parquet`, and `patient_condition_medication.parquet`.
- Milestone 7 final `baseline_scores.parquet` and
  `reports/milestone7_baseline_evaluation.json`.
- Milestone 8 train-fit `graph_edges.parquet` and
  `reports/milestone8_graph_suitability.json`.

Experiments:

- `xgboost_frozen_reference`
- `graph_only_xgboost`
- `xgboost_graph_augmented`
- `late_fusion_validation_weighted`
- `simple_ensemble_mean`

Selection uses MIMIC validation only. A graph-aware candidate is selected only
if it improves validation NDCG@10 by at least `0.005` absolute over frozen
XGBoost without dropping validation MRR@10 or Hit@10 by more than `0.01`.
Otherwise the frozen XGBoost reference remains selected.

## Commands

Lightweight synthetic verification:

```bash
uv run pytest tests/test_config.py tests/test_graph_ablation.py
```

Protected-data development run:

```bash
scripts/calculco/submit_graph_ablation.sh development
```

Protected-data final run after the frozen 8B selection exists:

```bash
scripts/calculco/submit_graph_ablation.sh final
```

## Acceptance Gates

- Milestone 7 final evaluation is complete and frozen.
- Milestone 8 graph suitability passed with a leakage audit status of `pass`.
- Graph features are derived only from MIMIC train-fit graph edges.
- Final MIMIC test metrics are blocked until `--mode final --frozen-selection`
  and `reports/milestone8b_frozen_selection.json` are present.
- eICU is reported as coverage-only unless in-catalog positives exist.
- Reports contain aggregate metrics only and no patient identifiers, row-level
  scores, raw concept samples, or clinical-note text.

## Related planning

Post-8B hybrid feature boundaries and branch-specific selection gates are
documented in `Documentation/HybridModelFeatureStrategy.md`. That document does
not change this milestone's scope or implement neural Transformer/GNN training.
# Milestone 8B Plan: Graph-Aware Ablation and Fusion Gate

## Summary
Milestone 8B should be implemented as a **graph-aware ablation milestone**, not a full Transformer-GNN build yet.

Evidence from review:
- Milestone 8 graph gate passed: `pass_for_graph_ablation`, 391,416 train-fit graph edges, 4,479 nodes, leakage audit passed.
- Milestone 7 final evaluation is recorded in `reports/milestone7_baseline_evaluation.json` with `mode=final`, `frozen_selection=true`.
- Frozen reference is `xgboost`; MIMIC test anchor is AP `0.1985`, ROC-AUC `0.7113`, Hit@10 `0.8481`, NDCG@10 `0.3699`, MRR@10 `0.4757`.
- eICU remains coverage-only because it has zero in-catalog positives.
- No Torch/PyG dependency exists, so a neural GNN/Transformer implementation should be gated behind evidence that graph signal improves over XGBoost.

## Key Changes
- Add config constants:
  - `MILESTONE8B_EVALUATION_ROOT = EVALUATION_ROOT / "milestone8b"`
  - `GRAPH_ABLATION_VERSION = "milestone8b-graph-ablation-v1"`
  - `MILESTONE8B_REPORT_VERSION = "milestone8b-ablation-evaluation-v1"`

- Add `pipeline.graph_ablation` CLI:
  - Inputs: Milestone 6 training/features, Milestone 7 baseline scores/report, Milestone 8 graph edges/reports.
  - Modes: `development` and `final`.
  - Development mode scores train diagnostics, MIMIC validation, and eICU coverage only.
  - Final mode requires `reports/milestone8b_frozen_selection.json` before MIMIC test metrics are emitted.
  - eICU metrics stay null/coverage-only unless in-catalog positives become evaluable.

- Build local ignored artifacts under `$DATASET_ROOT/processed/evaluation/milestone8b/`:
  - `graph_feature_matrix.parquet`
  - `graph_ablation_scores.parquet`
  - `models/graph_only_xgboost_model.json`
  - `models/graph_augmented_xgboost_model.json`
  - `models/graph_feature_preprocessor.joblib`
  - `fusion_weights.json`

- Write aggregate-only reports under `reports/`:
  - `milestone8b_graph_feature_manifest.json`
  - `milestone8b_ablation_evaluation.json`
  - `milestone8b_frozen_selection.json`

- Generate graph-derived candidate features from train-fit graph edges only:
  - direct condition-medication support and log support;
  - direct edge present flag;
  - condition degree/support summaries by relation type;
  - candidate medication graph degree/support summaries;
  - medication-medication coprescription degree/support summaries;
  - condition/candidate in-graph flags;
  - no patient identifiers, no row samples, no raw concept examples in reports.

- Evaluate these ablations:
  - `xgboost_frozen_reference`: existing Milestone 7 frozen score, no retraining.
  - `graph_only_xgboost`: graph-derived features only.
  - `xgboost_graph_augmented`: existing Milestone 7 feature family plus graph-derived features, same fixed XGBoost policy.
  - `late_fusion_validation_weighted`: validation-only weighted sum of frozen XGBoost score and graph-only score.
  - `simple_ensemble_mean`: 50/50 frozen XGBoost plus graph-only score.

- Selection rule:
  - Primary validation metric: NDCG@10.
  - Tie-breakers: MRR@10, Hit@10, AP, ROC-AUC, then simpler model.
  - Freeze a graph-aware candidate only if it improves validation NDCG@10 by at least `0.005` absolute over frozen XGBoost and does not reduce MRR@10 or Hit@10 by more than `0.01`.
  - Otherwise freeze `xgboost_frozen_reference` and record graph ablation as negative or inconclusive.

- Add Calculco wrappers:
  - `scripts/calculco/graph_ablation.sh`
  - `scripts/calculco/submit_graph_ablation.sh`
  - gitignore `scripts/calculco/milestone8b_job.env`
  - Use CPU-only OAR options, existing `common.sh`, bounded DuckDB temp/memory settings, and no GPU request.

- Documentation sync:
  - Update `README.md`, `Documentation/DataFoundationRoadmap.md`, `Documentation/Milestone8.md`, `WORKFLOWS.md`, `TESTING.md`, `CHANGELOG.md`, and `AGENT-MEMORY.md`.
  - Correct stale status saying Milestone 8 protected materialization is pending.
  - State clearly that 8B is graph-aware ablation, not validated clinical recommendation and not full Transformer-GNN deployment.

## Test Plan
- Add `tests/test_graph_ablation.py` with synthetic fixtures covering:
  - graph features are derived from train-fit graph edges only;
  - validation/test/eICU labels never affect graph statistics or fusion weights;
  - unseen condition/medication nodes produce cold-start flags and safe zero/null features;
  - graph-only, graph-augmented, and fusion scores have stable schemas;
  - final mode blocks MIMIC test metrics until frozen selection exists;
  - eICU remains coverage-only with zero positives;
  - reports contain no patient identifiers, row-level scores, or row samples;
  - selection rule freezes XGBoost when graph lift is below threshold.

- Verification commands:
  - `uv run pytest tests/test_config.py tests/test_graph_suitability.py tests/test_graph_ablation.py`
  - `uv run pytest tests/test_milestone7_baselines.py`
  - `uv run ruff check .`
  - `uv run ruff format --check .`

## Assumptions And Defaults
- No new runtime dependencies for 8B.
- No full Transformer, GNN, PyTorch, PyG, DDI, ontology, note, or rule-edge work in this milestone.
- MIMIC train is the only fitting source.
- MIMIC validation is the only model-selection source.
- MIMIC test is used only after 8B frozen selection.
- eICU remains external coverage-only until in-catalog positives exist.
- Observed prescriptions remain observational labels, not optimal treatment recommendations.
