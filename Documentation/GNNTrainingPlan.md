# GNN Training and Frozen-Transformer Hybrid Plan

## Summary

- Use this document as the canonical Phase D implementation and execution
  contract for `pipeline.gnn_training`.
- Treat Transformer v3 as immutable. Its frozen selection report, checkpoint, feature layout, calibration, and hashes become required inputs; GNN work must never overwrite or update them.
- Use the existing five node types, five train-fit relations, and patient subgraphs. Defer DDI, ontology, notes, patient-similarity edges, and eICU performance modeling.
- Implement a small native-PyTorch R-GCN-style branch rather than HGT or a new PyG dependency. R-GCN directly models typed relations, while HGT targets substantially larger and more complex heterogeneous graphs. Patient-specific graph pooling follows the relevant GraphCare design pattern. [R-GCN](https://arxiv.org/abs/1703.06103), [HGT](https://arxiv.org/abs/2003.01332), [GraphCare](https://openreview.net/pdf?id=tVTN7Zs0ml)
- Promotion outcome is binary: freeze a hybrid only if it beats the frozen Transformer gate; otherwise retain the Transformer and publish a negative aggregate result.

## Interfaces and Artifacts

- Add `pipeline.gnn_training` with these commands:
  - `prepare`: audit contracts, build typed vocabularies, compact the large patient-subgraph tables, and cache frozen Transformer representations.
  - `train-gnn` / `score-gnn`: train and qualify the independent relation branch.
  - `train-fusion` / `score-fusion`: fit late-fusion and residual joint-fusion candidates.
  - All scoring commands support `--mode development|final`; final mode requires `--frozen-selection`.
- Store restricted caches, checkpoints, predictions, and scores under `$DATASET_ROOT/processed/phase8_p0/gnn/`. Write only aggregate manifests and metrics under `$REPORTS_ROOT`.
- Preserve the existing canonical baseline-score schema so current DuckDB ranking metrics remain authoritative.
- Add aggregate reports for GNN preparation, standalone training/scoring/selection, hybrid training/scoring/selection, and frozen-artifact hashes.
- Use the separate CPU-prepare and GPU-training OAR wrappers under
  `scripts/calculco/`, both requiring `$WORK_SCRATCH`. No job was submitted as
  part of implementation.

## Implementation Status (2026-08-03)

- The five-stage CLI, native PyTorch relation model, fold-excluded graph
  preparation, frozen-Transformer representation cache, standalone GNN
  selection/refit/scoring, late/residual fusion, canonical scoring, and
  fail-closed development/final gates are implemented.
- Focused synthetic verification passes, including the complete
  `prepare → train-gnn → score-gnn → train-fusion → score-fusion` workflow.
  This is software verification only; it is not evidence of clinical utility.
- Protected `prepare` and fold-excluded cross-fit cache construction completed.
  Development GPU job 8825 then failed in `train-gnn`, before model
  optimization, because the loader required one Parquet file per logical edge
  partition while DuckDB emitted multiple fragments. The loader now combines
  all fragments in deterministic filename order while retaining one logical
  shard as the memory boundary. No protected GNN/fusion model or metric is
  claimed until training and scoring are rerun.
- Cross-fit preparation currently materializes five physical fold-excluded
  cache trees. Because this is storage-intensive, protected preparation fails
  unless `GNN_CROSSFIT_MIN_FREE_GIB` records a reviewed threshold and the
  target filesystem has at least the larger of that threshold and the
  cache-derived estimate. The CPU OAR wrapper also requires DuckDB spill under
  `$WORK_SCRATCH`, defaults to `DUCKDB_THREADS=4` / `DUCKDB_MEMORY_LIMIT=128GB`
  on 512 GB CPU nodes, and rebuilds forward edges one ranking-group shard at a
  time to avoid a full `MATERIALIZED` scan of `subgraph_edges`.
- The Transformer representations used for fusion are from its immutable
  full-train refit, not a Transformer cross-fit. Therefore the late-fusion
  weight is explicitly an in-sample train meta-fit over GNN OOF logits plus a
  fixed train-derived Transformer covariate. It is not reported as complete
  hybrid OOF evidence; the separate MIMIC validation gate remains the
  promotion evidence.

## Implementation and Training

1. **Immutable contract gate**
   - Require the existing training-contract lock, completed graph/subgraph manifests, and frozen Transformer selection with `status=frozen` and `model_frozen=true`.
   - Verify hashes for the Transformer checkpoint, calibration, and feature layout before every stage.
   - Load the Transformer with `eval()`, `requires_grad=False`, `no_grad()`, and detached outputs. Tests must prove no Transformer tensor receives gradients or changes on disk.
   - Lock `temporal-features-v2`, `graph-suitability-v1`, `observed-medication-label-v1`, and `patient-split-v1`.

2. **Memory-bounded graph preparation**
   - Project only approved fields from nodes, edges, candidates, and the subgraph index; encode strings into train-derived integer vocabularies and omit identifiers from model tensors.
   - Hash complete ranking groups into 256 compact Parquet shards. Never load or scan the 1.2-billion-edge artifact globally during training.
   - Preserve every candidate in each group. Exclude zero-positive groups only from fitting; retain and report them during evaluation.
   - Expand the five relations with deterministic reverse relations and one self-loop relation.
   - Transform support as `log1p(support_count)` and normalize incoming weights within each relation and destination node.
   - Cache the frozen 256-dimensional Transformer stay-context vector and frozen candidate logits, keyed locally by restricted group identifiers. Record the Transformer hash in every cache manifest.

3. **Standalone GNN**
   - Node representation: unified concept identity embedding plus node-type and node-role embeddings and the existing observed/cold-start flags.
   - Encoder: two 128-dimensional relation-aware message-passing layers with residual connections, LayerNorm, GELU, and dropout `0.2`.
   - Candidate representation: concatenate the query-condition node, candidate-medication node, their elementwise interaction, and attention-pooled observed context nodes.
   - Score with a two-layer MLP; include `log1p(candidate_rank)` because the frozen Stage 1 selection retained that prior.
   - Pre-register four train-fold comparisons: full five relations, no message passing, no direct condition–medication relation, and no lab/vital/intervention relations. Use deterministic patient-grouped MIMIC-train folds. For each held-out fold, exclude its patients before relation support, coprescription, temporal event edges, concept vocabulary, cold-start flags, and normalized edge weights are fitted. The official full-train graph is refit/reference-only.
   - Optimize the existing multi-positive listwise loss plus primary-positive and auxiliary BCE terms, using AdamW (`3e-4`, weight decay `1e-4`), gradient clipping `1.0`, mixed precision, maximum 30 epochs, patience 3, and seed `20260617`.
   - Refit the selected variant on all MIMIC train data using the median fold-selected epoch. Evaluate MIMIC validation once.
   - Qualify the branch only if it improves NDCG@10 by at least `0.005` over graph-only XGBoost without reducing MRR@10 or Hit@10 by more than `0.01`.

4. **Frozen-Transformer fusion**
   - Fit a late-fusion baseline using a single constrained weight over
     within-group normalized frozen-Transformer and GNN logits. The GNN side is
     patient-level OOF; the Transformer side is the fixed full-train-refit
     representation described in the implementation-status limitation above.
   - Fit a residual hybrid initialized to reproduce the Transformer exactly:
     `hybrid_logit = frozen_transformer_logit + residual_mlp(transformer_context, gnn_candidate_representation)`.
   - Initialize the residual output layer to zero. Train only the GNN copy and fusion head; the standalone GNN and Transformer checkpoints remain unchanged. Use the frozen standalone-GNN fold-selected epoch count for the full-train residual refit instead of selecting residual epochs from contaminated train-fold Transformer outputs.
   - Select between late fusion and residual fusion using validation NDCG@10, then MRR@10, Hit@10, and the simpler model.
   - Read the reference dynamically from the frozen Transformer selection. With the current report, the hybrid threshold is approximately NDCG@10 `0.417865`, with MRR@10 and Hit@10 floors of approximately `0.516109` and `0.869446`.
   - If the gate passes, freeze all hashes, fit temperature calibration after ranking selection, and allow one final MIMIC-test scoring run. Otherwise block final hybrid scoring and retain the Transformer.
   - Run two additional seeds only as stability analyses after the primary run; they cannot replace the preselected seed.
   - Report paired patient-bootstrap confidence intervals for metric deltas as supporting evidence, while keeping the existing point-lift gate authoritative.

## Verification and Acceptance

- Synthetic tests must cover:
  - frozen-Transformer hash drift, gradient blocking, and overwrite prevention;
  - train-only graph/vocabulary fitting and temporal-cutoff enforcement;
  - complete-group sharding, bounded shard reads, local-index offsets, reverse edges, self-loops, relation mapping, and support normalization;
  - empty-edge, cold-start, OOV, zero-positive, multi-positive, and variable-size graphs;
  - batched versus single-graph score parity;
  - relation-ablation behavior and GNN/fusion gradient ownership;
  - canonical score-schema and ranking-metric parity;
  - development/final gating and frozen-artifact drift;
  - aggregate-report safety with no patient identifiers, row samples, notes, or raw concepts;
  - end-to-end `prepare → train-gnn → score-gnn → train-fusion → score-fusion`.
- Run focused tests, the full suite, Ruff check, Ruff format check, configuration parsing, and final diff review exclusively through `uv`.
- Protected completion remains pending and requires reviewed aggregate
  manifests, successful authorized OAR runs, unchanged frozen Transformer
  hashes, untracked restricted artifacts, and a frozen validation decision.

## Assumptions and Boundaries

- Defaults chosen after unanswered planning prompts: train the GNN and then fusion; use the existing graph; create a new canonical plan document.
- No new runtime dependency is added: use the existing optional PyTorch group. PyG requires a later profiling-based justification.
- External DDI/contraindication edges are deferred because meaningful safety optimization requires a curated, versioned source; do not report DDI safety from the current graph. [SafeDrug](https://www.ijcai.org/proceedings/2021/514)
- eICU remains coverage-only for the current RxNorm-first graph. ATC-3 external evaluation requires a separately reviewed artifact rebuild.
- Observed prescriptions remain historical labels and unobserved candidates remain weak negatives; no output is described as optimal treatment or validated clinical advice.
- MIMIC test has already been scored for the frozen Transformer. Hybrid test scoring remains final-only, but reports must disclose that the overall test split is not wholly unseen to the broader research program.
- Preserve all current uncommitted Transformer-v3 work and avoid unrelated changes.
