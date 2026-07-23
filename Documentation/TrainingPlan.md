# Gate-First Training Plan For Phase 8 P0 Models

## Implementation Status

- Stage 1 code is implemented in `pipeline.training_contract` and
  `pipeline.gate_recovery`, with a CPU-only Calculco wrapper and synthetic
  contract/ranking tests.
- The protected-data contract audit and recovery experiment have not been
  submitted. Until that run clears the gate, frozen `xgboost_frozen_reference`
  remains canonical and MIMIC test scoring remains blocked.
- Stage 2 is intentionally not implemented: PyTorch has not been added and no
  neural cache, model, or GPU wrapper exists.
- Feature promotion compares with the prior canonical graph model. Neural
  readiness compares with Phase 8 P0 `xgboost_frozen_reference`: validation
  NDCG@10 is `0.374899`, making the six-decimal pass target `0.379899`.

## Summary
- Use the completed `processed/phase8_p0/` package as the immutable training input contract, pinned to `temporal-features-v2`, `graph-suitability-v1`, `observed-medication-label-v1`, and `patient-split-v1`.
- Do not begin full neural Transformer-GNN training yet. Current Step 10 evidence says `proceed_to_neural_transformer_gnn=false`; frozen `xgboost_frozen_reference` remains selected because graph lift did not clear +0.005 NDCG@10.
- Training proceeds in two gates: first improve/validate structured feature and graph ablations against frozen XGBoost; only then add neural branch training.

## Key Changes
- Freeze baseline anchors from Phase 8 P0:
  - Deterministic patient-grouped MIMIC-train folds select all recovery
    features, hyperparameters, and fusion weights.
  - MIMIC validation is evaluated once for the locked recovery gate.
  - MIMIC test is final-only after frozen selection.
  - eICU primary `rxnorm_or_atc` remains coverage-only; use `atc3_or_rxnorm` only for external sensitivity metrics.
- Add a future `pipeline.neural_training` surface only after the gate clears:
  - Inputs: `patient_stay_features`, `event_sequences`, `patient_condition_medication`, vocabularies, train-fitted preprocessor, `graph_edges`, and `patient_subgraphs`.
  - Outputs: ignored local model/cache artifacts under `$DATASET_ROOT/processed/phase8_p0/neural/` and aggregate-only reports under `reports/`.
- Model sequence:
  - Re-run and strengthen tabular/graph ablations first, keeping XGBoost as the reference.
  - If gate passes, train branch models: Transformer patient/context branch, GNN relation branch, then late-fusion and joint-fusion rankers.
  - Use group-wise ranking batches keyed by `ranking_group_id`; score all candidate medications in each `stay + condition` group.
- Neural objective after gate:
  - Use multi-positive group softmax loss for groups with observed positives.
  - Track BCE/calibration diagnostics separately.
  - Optimize validation NDCG@10; require at least +0.005 absolute over frozen XGBoost with no MRR@10 or Hit@10 drop greater than 0.01.

## Training Flow
- Phase A: Contract audit
  - Confirm all Step 9 artifacts and aggregate manifests are complete.
  - Reconfirm train-only graph/vocab/preprocessor fitting and patient-level split integrity.
- Phase B: Gate recovery
  - Run targeted structured ablations before neural work: condition feature
    caps, graph support thresholds, graph-feature transformations,
    candidate-rank ablation, and train out-of-fold fusion.
  - Freeze a new candidate only if it clears the existing 8B lift rule.
- Phase C: Neural branch smoke tests
  - Build tiny synthetic and bounded MIMIC-train loaders without raw-row reporting.
  - Train Transformer-only and GNN-only models separately; compare each to XGBoost and graph-only baselines.
- Phase D: Hybrid training
  - Fuse patient embedding, sequence embedding, candidate medication embedding, and GNN subgraph embedding.
  - Run validation selection, freeze once, then emit final MIMIC test and ATC-3 eICU sensitivity reports.
- Phase E: Reporting
  - Report aggregate metrics only: NDCG@10, MRR@10, Hit@10, precision@10, recall@10, AP, ROC-AUC, Brier score, ECE, coverage, cold-start rates, and label caveats.

## Test Plan
- Add synthetic tests for neural dataset readers, group batching, temporal cutoff enforcement, train-only vocab use, no validation/test/eICU graph fitting, and no patient identifiers in reports.
- Stage 1 tests cover contract failures, safe projection, patient-fold
  isolation, positive-group sampling, metric parity, changed-lock detection,
  and final-mode blocking. Neural loader/loss tests remain conditional on a
  Stage 1 pass.
- Add metric-parity tests against existing ranking metric functions.
- Add final-mode gating tests so test metrics cannot run before frozen validation selection.
- Verification commands:
  - `uv run pytest tests/test_model_ready_package.py tests/test_patient_subgraphs.py`
  - `uv run pytest tests/test_milestone7_baselines.py tests/test_graph_ablation.py`
  - future: `uv run pytest tests/test_neural_training.py`
  - `uv run ruff check .`
  - `uv run ruff format --check .`

## Assumptions
- Default direction is gate-first because no override was confirmed.
- No notes, DDI/ontology edges, patient-similarity graphs, or pooled MIMIC+eICU training.
- Observed labels remain historical prescribing in `(24h, 48h]`, not optimal treatment.
- Neural work is research-only and cannot be described as a validated clinical recommender.
