# Gate-First Phase 8 P0 Training Implementation

## Summary

- Preserve [TrainingPlan.md](/nfs/home/lisic/zahmed/ResearchModule/Documentation/TrainingPlan.md) as the governing direction, with these comparisons:
  - Feature promotion / Stage 1 recovery compares against Phase 8 P0
    `xgboost_frozen_reference` (validation NDCG@10 `0.374899`; Stage 1 pass
    target `0.379899`).
  - Stage 2 neural readiness compares against the frozen Stage 1 recovery
    winner (`xgboost_rank_ndcg_oof_late_fusion`, validation NDCG@10 ≈
    `0.394607`; neural pass target ≈ `0.399607`).
- Stage 1 contract audit and rank-aware recovery are implemented; the
  protected development gate has passed and authorized neural training.
- Keep all patient-level artifacts in protected ignored storage and all reports aggregate-only.

## Stage 1: Gate Recovery

- Add a training-contract audit that validates:
  - Exact four pinned versions and completed upstream manifests.
  - MIMIC patient-level split integrity.
  - Train-only graph, vocabulary, preprocessing, and candidate fitting.
  - The 24-hour cutoff, `(24h,48h]` label window, and medication-free event sequences.
  - Artifact schemas, row counts, file metadata, and manifest hashes without reading or reporting patient rows.
- Reject unsafe model columns, including identifiers, source event IDs, `source_text`, `value_text`, raw codes, hospital/ward identifiers, outcome fields, and provenance-only columns.
- Write an aggregate `phase8_p0_training_contract_lock.json`; all later runs must match it.

- Add a rank-aware recovery runner reusing existing metric and report machinery:
  - Use deterministic three-fold patient grouping inside MIMIC train for all feature, hyperparameter, and fusion decisions.
  - Exclude groups with no observed in-catalog positive from fitting; report them as coverage exclusions.
  - Keep every positive and deterministically sample up to `max(10, 5 × positive_count)` negatives within each ranking group.
  - Use XGBoost `rank:ndcg`, `ndcg@10`, early stopping of 50 rounds, and a maximum of 1,000 rounds.
  - Screen condition caps `{0, 20, 40}`, graph support thresholds `{1, 5, 10}`, graph families `{direct, context, all}`, and candidate-rank inclusion `{on, off}` in that order.
  - Recompute all graph summaries consistently after support filtering.
  - Tune only four locked configurations: depth `{4,6}` crossed with `(learning_rate, min_child_weight)` `{(0.05,10), (0.03,1)}`; retain subsample and column sampling at `0.8`.
  - Select by mean train-fold NDCG@10, then MRR@10, Hit@10, fewer features, and stable experiment name.
  - Select any late-fusion weight from train out-of-fold scores using the existing `0.05` grid; never tune it on MIMIC validation.

- Refit the single locked candidate on all MIMIC train data and evaluate MIMIC validation once.
- Pass only when:
  - NDCG@10 improves by at least `0.005` over Phase 8 P0 frozen XGBoost.
  - MRR@10 and Hit@10 each drop by no more than `0.01`.
- If it fails, emit a negative aggregate report, retain XGBoost, and stop before adding neural dependencies.
- Final MIMIC test scoring remains blocked until a passing selection report exists.

## Stage 2: Conditional Neural Prototype

- After a Stage 1 pass, add an optional `neural` dependency group containing PyTorch; do not add PyG initially.
- Provide `prepare`, `train`, and `score` commands under one `pipeline.neural_training` interface with `development` and `final` modes.
- Store tensor caches, checkpoints, predictions, and calibration parameters under `$DATASET_ROOT/processed/phase8_p0/neural/`; store only aggregate manifests in `reports/`.

- Prepare hash-sharded, stay-grouped training caches:
  - Project only approved columns.
  - Encode each stay’s context once and score all of its condition groups.
  - Reserve `PAD=0` and `UNK=1`, offsetting train-derived vocabulary indexes by two.
  - Keep the most recent `{64,128,256}` events using stable timestamp/sequence ordering; select length using MIMIC-train folds.
  - Fit numeric normalization and per-event-token value statistics on MIMIC train only.
  - Preserve all candidates in positive ranking groups.
  - Stream patient subgraphs by shard; never load the 1.2-billion-edge table globally.

- Implement fixed initial architectures:
  - Transformer branch: 128-dimensional event embeddings, two pre-norm encoder layers, four heads, 256-dimensional feed-forward blocks, dropout `0.1`, and an approved tabular-context MLP.
  - GNN branch: two relation-aware message-passing layers over the five node and five relation types, reverse edges and self-loops added deterministically, `log1p(support_count)` edge weights, and candidate-specific query pooling.
  - Fusion: first evaluate train-fold late fusion, then a joint MLP over patient, condition, candidate, sequence, and GNN embeddings; include candidate rank only if its Stage 1 ablation retained it.

- Train with:
  - Multi-positive listwise softmax plus auxiliary BCE at weight `0.25`.
  - AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient clipping `1.0`, mixed precision where available, maximum 30 epochs, and patience 3.
  - Primary seed `20260617`; two additional seeds are stability analyses and cannot replace the preselected run.
  - Temperature calibration fitted from train out-of-fold predictions; rank using logits and compute Brier/ECE from calibrated sigmoid probabilities.

- Evaluate only the locked Transformer on MIMIC validation. Freeze it only when
  it clears +0.005 NDCG@10 over the Stage 1 recovery winner with the same
  secondary-metric guardrails; only then score MIMIC test.
- Treat eICU as:
  - Coverage-only for primary `rxnorm_or_atc`.
  - A separately trained, architecture-frozen `atc3_or_rxnorm` sensitivity model with its own MIMIC-train graph, vocabularies, and subgraphs.
  - Never use eICU for tuning or pooled fitting.

## Interfaces and Verification

- Every score artifact must preserve the existing baseline score schema so current ranking metrics remain authoritative.
- Selection manifests must record package-lock digest, git revision, seed, folds, feature families, hyperparameters, epochs, calibration, cohort/window semantics, and observational-label caveat.
- Add synthetic tests for contract failures, safe column projection, patient-fold isolation, group-preserving sampling, OOV handling, sequence truncation, train-only normalization, graph fit scope, multi-positive loss, zero-positive exclusion, metric parity, report safety, and final-mode blocking.
- Run focused tests, the complete suite, Ruff check, and Ruff format check through `uv`.
- Add CPU-only OAR wrappers for Stage 1 and configurable GPU wrappers for Stage 2, using `$WORK_SCRATCH` for temporary I/O and aggregate-only logs.
- Synchronize architecture, roadmap, workflow, testing, README, changelog, and training-plan status after each completed stage.

## Assumptions

- The attached gate-first direction is confirmed; no research override is inferred.
- Notes, external DDI/ontology edges, patient-similarity graphs, and pooled MIMIC/eICU training remain excluded.
- Observed prescribing remains a historical label, not an optimal-treatment claim.
- Implementation may add and verify wrappers, but no long-running OAR or GPU job is submitted without explicit authorization.
