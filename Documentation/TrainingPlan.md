# Gate-First Training Plan For Phase 8 P0 Models

## Implementation Status

- Stage 1 code is implemented in `pipeline.training_contract` and
  `pipeline.gate_recovery`, with a CPU-only Calculco wrapper and synthetic
  contract/ranking tests.
- The protected-data Stage 1 development run completed and froze
  `xgboost_rank_ndcg_oof_late_fusion` (validation NDCG@10 ≈ `0.394607`). The
  selection report records `neural_training_authorized=true` and
  `decision=promote_to_neural_prototype`. Optional Stage 1 `final` (held-out
  MIMIC test) has not been required for neural authorization.
- Stage 2 Transformer patient/context training is implemented in
  `pipeline.neural_training` (Phase C, Transformer-only branch): a
  `prepare`/`train`/`score` CLI, DuckDB cache preparation, a PyTorch Transformer
  recommender, listwise + auxiliary training with early stopping and temperature
  calibration, canonical-schema scoring, a neural gate/selection report, an
  optional `neural` PyTorch dependency group, a GPU OAR wrapper
  (`scripts/calculco/phase8_p0_neural_training.sh`), and synthetic
  torch-guarded tests. Every stage is fail-closed behind the Stage 1 gate; the
  GNN branch and joint fusion (Phase D) are not implemented yet.
- Experiment version `phase8-p0-neural-transformer-v2` (gap recovery) adds a
  numeric MLP encoder, learned positional encodings, dual-path candidate
  scorer, train-only global/condition×candidate priors plus `log1p(candidate_rank)`,
  warmup+cosine AdamW (`1e-4` / WD `1e-3` / dropout `0.2` / patience 5), and
  per-batch aux-BCE `pos_weight`. Rerun `prepare` before `train` so group
  caches include the new candidate-side columns.
- Stage 2 on protected data is authorized by the frozen Stage 1 selection. The
  Transformer must beat that Stage 1 winner (not the older Milestone 8B
  `xgboost_frozen_reference` at `0.374899`): required validation NDCG@10 is at
  least ≈ `0.399607` (+0.005), with MRR@10 / Hit@10 drops of at most `0.01`.
  The first protected development run (`transformer-v1`) completed but failed
  the neural gate (`retain_structured_recovery_baseline`; best epoch 0,
  NDCG@10 ≈ `0.3747`).

## Summary
- Use the completed `processed/phase8_p0/` package as the immutable training input contract, pinned to `temporal-features-v2`, `graph-suitability-v1`, `observed-medication-label-v1`, and `patient-split-v1`.
- Stage 1 structured recovery has cleared the Milestone 8B XGBoost bar and
  authorized the Transformer prototype. Full Transformer-GNN / joint fusion
  remains later work.
- Training still proceeds in gates: Stage 1 structured recovery against frozen
  Milestone 8B XGBoost; Stage 2 neural branch against the Stage 1 recovery
  winner.

## Key Changes
- Freeze baseline anchors from Phase 8 P0:
  - Deterministic patient-grouped MIMIC-train folds select all recovery
    features, hyperparameters, and fusion weights.
  - Configuration screening compares candidates on the identical bounded train
    sample; the locked finalist, binary reference OOF, and validation gate use
    the full candidate universe. Finalist and reference out-of-fold scores stream
    to narrow Parquet per fold and the fusion-weight search runs in DuckDB, so
    peak memory stays bounded (earlier full-universe pandas fusion was
    OOM-killed).
  - MIMIC validation is evaluated once for the locked recovery gate.
  - MIMIC test is final-only after frozen selection.
  - eICU primary `rxnorm_or_atc` remains coverage-only; use `atc3_or_rxnorm` only for external sensitivity metrics.
- The `pipeline.neural_training` surface is implemented and Stage 1-authorized
  (Transformer branch only so far):
  - Inputs: `patient_stay_features`, `event_sequences`, `patient_condition_medication`, and train-derived vocabularies. The GNN inputs `graph_edges` and `patient_subgraphs` are reserved for the not-yet-implemented GNN/fusion branch.
  - Outputs: ignored local model/cache artifacts under `$DATASET_ROOT/processed/phase8_p0/neural/` and aggregate-only reports under `reports/` (`phase8_p0_neural_prepare_manifest.json`, `phase8_p0_neural_training_evaluation.json`, `phase8_p0_neural_score_evaluation.json`, `phase8_p0_neural_training_selection.json`).
  - Gate: compare against Stage 1 recovery winner scores
    (`.../evaluation/gate_recovery/baseline_scores.parquet`, default baseline
    `xgboost_rank_ndcg_oof_late_fusion`).
- Model sequence:
  - Re-run and strengthen tabular/graph ablations first, keeping XGBoost as the reference.
  - If gate passes, train branch models: Transformer patient/context branch, GNN relation branch, then late-fusion and joint-fusion rankers.
  - Use group-wise ranking batches keyed by `ranking_group_id`; score all candidate medications in each `stay + condition` group.
- Neural objective after gate:
  - Use multi-positive group softmax loss for groups with observed positives.
  - Track BCE/calibration diagnostics separately.
  - Optimize validation NDCG@10; require at least +0.005 absolute over the
    Stage 1 recovery winner (`xgboost_rank_ndcg_oof_late_fusion`) with no
    MRR@10 or Hit@10 drop greater than 0.01.

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
  and final-mode blocking.
- Stage 2 neural tests are implemented: `tests/test_neural_data.py`,
  `tests/test_neural_metrics.py`, and `tests/test_neural_contract.py` are
  torch-free; `tests/test_neural_dataset.py`, `tests/test_neural_model.py`, and
  `tests/test_neural_train_score.py` skip without the optional `neural` group.
  They cover dataset readers, group batching, temporal cutoff enforcement,
  train-only vocab use, the fail-closed preflight/gate, ranking-metric parity,
  and an end-to-end prepare/train/score smoke test.
- Verification commands:
  - `uv run pytest tests/test_model_ready_package.py tests/test_patient_subgraphs.py`
  - `uv run pytest tests/test_milestone7_baselines.py tests/test_graph_ablation.py`
  - `uv run pytest tests/test_neural_data.py tests/test_neural_metrics.py tests/test_neural_contract.py`
  - `uv sync --group neural && uv run pytest tests/test_neural_dataset.py tests/test_neural_model.py tests/test_neural_train_score.py`
  - `uv run ruff check .`
  - `uv run ruff format --check .`

## Assumptions
- Default direction is gate-first because no override was confirmed.
- No notes, DDI/ontology edges, patient-similarity graphs, or pooled MIMIC+eICU training.
- Observed labels remain historical prescribing in `(24h, 48h]`, not optimal treatment.
- Neural work is research-only and cannot be described as a validated clinical recommender.
