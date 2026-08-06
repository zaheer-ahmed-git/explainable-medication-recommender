## Verdict

The current GNN job is not deadlocked, but it is unhealthy and very unlikely to finish the four-stage workflow within its 48-hour allocation. I recommend cancelling it after preserving the existing logs; restarting unchanged would repeat the same structural problems.

At the latest review:

- OAR still reported job `12214` as running at 03:30 CEST.
- A live GPU snapshot showed 77% utilization and about 26 GiB used on an A100 40 GB, so computation was still occurring.
- There were still no GNN checkpoints or training-state files.
- The public training report remained the stale job `10962` failure.
- The CUDA allocator warning in the [latest stderr](/nfs/home/lisic/zahmed/ResearchModule/scripts/calculco/logs/rm_phase8_p0_gnn_training_12214.err:3) occurred approximately 7h49m after job start.

The [verdict note](/nfs/home/lisic/zahmed/ResearchModule/scripts/calculco/logs/jobverdictgnn.md:1) correctly identified the OOM risk, but “reduce batch size” addresses only one symptom. The deeper causes are the message-passing implementation, multi-hour integrity preflight, fragmented cache layout, and non-resumable serial workflow.

## Ranked root causes

| Priority | Root cause | Confidence |
|---|---|---|
| P0 | Per-edge dense relation matrices create a ~5.5 GiB temporary | Near certain |
| P0 | Exact artifact verification re-hashes 225,945 files several times per stage | Very high |
| P0 | Four variants × five folds run serially without epoch checkpoints or resume | Very high |
| P1 | Fragmented Parquet/Pandas loader adds substantial metadata and CPU overhead | High |
| P1 | FP16 is a poor precision choice for the current aggregation path | High |
| P2 | Graph representation is dominated by global priors and binary concept presence | Moderate/high |

### 1. The CUDA OOM matches one exact tensor operation

The model has only 1,071,943 parameters; model size is not the problem.

The problem is [model.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/gnn_training/model.py:163):

```python
transforms = self.relation_weight[edge_type]
messages = torch.einsum("ei,eij->ej", source_states, transforms)
```

This materializes an `E × 128 × 128` relation matrix tensor. The protected fold manifest records approximately 1.433 billion expanded edges across 514,083 groups—about 2,788 edges per group. At the configured batch size of 32:

- Expected edges per batch: approximately 89,219.
- FP32 relation tensor: approximately 5.45 GiB.
- Failed allocation: approximately 5.54 GiB.
- The failed allocation corresponds to approximately 90,752 edges.

That is an exceptionally close match. AMP does not solve it because indexing the FP32 relation parameter produces the large FP32 temporary before the autocast `einsum`.

### 2. Preflight consumes most of the silent startup

The cross-fit tree contains 225,945 files and has an estimated size of approximately 132 GB, recorded in the [cross-fit manifest](/nfs/home/lisic/zahmed/ResearchModule/reports/phase8_p0_gnn_crossfit_graph_manifest.json:5).

For every preflight, the verifier:

1. Hashes every recorded file individually.
2. Re-hashes each complete fold tree.
3. Re-hashes the entire cross-fit tree again.

See [crossfit.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/gnn_training/crossfit.py:2153). All four training/scoring stages require this verification.

This explains the repeated pattern:

- Hours of silence.
- First CUDA/model failure only after roughly 7–8 hours.
- Old reports appearing to have start-time timestamps because `generated_at` is captured before preflight.
- Approximately three full-tree reads per stage, potentially twelve across the complete four-stage chain.

The integrity checks are clinically and scientifically appropriate, but their implementation is operationally prohibitive.

### 3. The workflow cannot reliably fit into 48 hours

[train_gnn.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/gnn_training/train_gnn.py:174) executes four variants over five folds serially: 20 complete fits, each with up to 30 epochs and a full held-out evaluation after every epoch.

Additional problems:

- A fold checkpoint is written only after the fold finishes, not after each epoch.
- There is no completed-fold skip or exact resume.
- The running report is held in memory and written only on failure/completion.
- The OAR wrapper then runs scoring, fusion training, and fusion scoring in the same job, each with another expensive preflight; see [phase8_p0_gnn_training.sh](/nfs/home/lisic/zahmed/ResearchModule/scripts/calculco/phase8_p0_gnn_training.sh:122).

Even if the current job survives memory pressure, the complete chain is not realistically schedulable in its present form.

### 4. Cache fragmentation and loading are inefficient

The loader now correctly handles the multi-file layout that broke job `8825`, but it does so by opening and concatenating every fragment with Pandas for every logical shard and every epoch; see [dataset.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/gnn_training/dataset.py:888).

It then performs Python group-by reconstruction, sorting, validation, collation, and tensor copies without:

- background workers;
- pinned memory;
- asynchronous host-to-device transfer;
- prefetching;
- persistent reconstructed examples.

The allocation reserves 48 CPU cores, while the observed Python process used only a fraction of one core during the snapshot. Much of the CPU allocation is therefore ineffective.

### 5. Mixed precision handling is improved but incomplete

The loss-scale retry patch correctly fixes the specific job `10962` behavior. Gradient clipping was the detector, not the cause.

Remaining concerns:

- CUDA autocast defaults to FP16 even though the A100 supports BF16.
- Loss-scale backoff handles backward overflow, not non-finite forward activations.
- Twenty-four retries can repeat an expensive graph forward/backward many times.
- Dropout changes between retries, so “same batch retry” is not exactly the same computation.
- Overflow, loss-scale, memory, and batch-shape information is not emitted until an epoch completes.

BF16 or FP32 aggregation should be preferred over relying on increasingly small FP16 loss scales.

## Architecture and research-method review

The leakage controls are generally strong:

- Patient-grouped folds are enforced.
- Held-out patients are removed before support, coprescription, vocabulary, cold-start, and edge-normalization fitting.
- The full-train cache is explicitly not selection eligible.
- Zero-positive groups are excluded from fitting but retained for coverage/evaluation.
- No patient-level rows were exposed during this review.

However, the current GNN signal is limited:

- Patient context nodes encode concept presence only. Numeric lab/vital values, abnormality, magnitude, trend, and recency are discarded in [patient_subgraphs.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/patient_subgraphs.py:362).
- There is no explicit patient/stay node.
- Condition–vital edges are almost complete and condition–lab edges are dense, making them weakly selective and prone to oversmoothing.
- Direct condition–medication edges and `candidate_rank` both encode train-derived prescribing priors. They may dominate patient-specific graph evidence.
- Coprescription edges represent historical co-use, not DDI safety or optimal treatment.

Two methodological limitations also deserve correction:

- Standalone GNN temperature is fitted on validation before the validation qualification run; see [train_gnn.py](/nfs/home/lisic/zahmed/ResearchModule/pipeline/gnn_training/train_gnn.py:245). Ranking is unaffected by monotonic temperature scaling, but validation calibration metrics are optimistic.
- Late fusion uses GNN OOF predictions with full-train-refit Transformer predictions. This limitation is disclosed correctly, but it is not complete hybrid OOF evidence.

Documentation is inconsistent: [ARCHITECTURE.md](/nfs/home/lisic/zahmed/ResearchModule/ARCHITECTURE.md:45) says GNN/fusion remain planned, while later sections describe them as implemented; [WORKFLOWS.md](/nfs/home/lisic/zahmed/ResearchModule/WORKFLOWS.md:604) still says protected preparation has never run.

## Prioritized action plan

### P0 — before another full run

1. Cancel the current full-chain attempt. It is producing useful telemetry but has no resumable state and cannot finish the registered workflow reliably.

2. Replace the per-edge relation-matrix expansion with a mathematically equivalent relation-grouped implementation:

   - For each of the 11 relations, aggregate weighted source states with `index_add_`.
   - Apply the relation’s single `128 × 128` matrix after aggregation.
   - Never create `E × 128 × 128`.

3. Batch by edge/node budget, not group count. Until the operator is fixed, use 1–2 groups only. Afterwards select a `max_edges_per_batch` from profiling and use gradient accumulation to preserve an effective 32-group batch.

4. Add explicit precision modes: `fp32`, `bf16`, and `fp16`. Use BF16 on the A100, with message aggregation, LayerNorm, softmax/loss, and gradient-norm calculation retained in FP32 where necessary.

5. Compact the cache to approximately one file per table/split/shard. Preserve exact integrity by producing a new compacted-tree manifest and hashes.

6. Verify the tree once per allocation and issue a locked preflight attestation. Subsequent stages should verify the small attestation and immutable manifest rather than reading the entire tree repeatedly.

7. Split execution into resumable stages:

   - OAR array for each `(variant, fold)`.
   - Per-epoch model/optimizer/scaler/RNG checkpoint.
   - Completed-fold manifest with config and input hashes.
   - Separate selection/refit, scoring, and fusion jobs with scheduler dependencies.

8. Emit aggregate-safe heartbeats every N batches: variant, fold, epoch, processed groups/edges, throughput, loss scale, retries, allocated/reserved/peak CUDA memory, and elapsed time.

### P1 — representation and modeling

- Add a rank-only/no-graph baseline explicitly, not only `no_message_passing`.
- Add learned relation gates or relation dropout; test removal of dense vital/lab relations.
- Add an explicit stay/query node connected to observed context.
- Encode numeric value, abnormality direction, trend, and time bin as edge/node attributes.
- Consider encoding the small fold-global concept graph once and pooling patient-observed concept embeddings instead of duplicating more than a billion induced edges.
- Fit temperature from train OOF predictions or a dedicated calibration split.
- Do not alter learning rate or other hyperparameters until memory and precision problems are isolated. `3e-4`, clipping at `1.0`, and AdamW are not currently the primary suspects.

## Required diagnostics before restart

A restart should require all of these gates:

1. Production-shaped one-batch profiler on the A100 for batch sizes 1/2/4/8:

   - peak allocated/reserved memory;
   - edges and nodes per batch;
   - shard load, collation, H2D, forward, backward, and optimizer timings.

2. Numerical matrix across FP32, BF16, and FP16:

   - finite logits/loss/gradients;
   - per-parameter aggregate gradient norms;
   - AMP retry count and minimum scale;
   - new versus old operator output/gradient parity on small graphs.

3. Edge-count distribution, including p50/p95/p99/max, so one unusually large graph cannot OOM a fixed-group batch.

4. One-fold/one-variant/one-epoch smoke run proving:

   - heartbeat output;
   - epoch checkpoint creation;
   - exact resume after controlled interruption;
   - no patient identifiers in reports.

5. One complete fold to early stopping, followed by a walltime extrapolation. Only schedule all 20 fits if the estimate fits the chosen OAR-array plan with margin.

6. After software stability, run the pre-registered primary seed. Additional seeds remain stability analyses and must not replace it.

## Verification

- All 44 focused GNN tests passed in 34.73 seconds.
- Focused Ruff checks passed.
- No repository files were changed.
- The only untracked file remains the user-provided GNN verdict note.
- No clinical rows or patient identifiers were inspected or reported.