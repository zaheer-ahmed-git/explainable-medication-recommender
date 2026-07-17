# Phase 8 P0 Model-Ready Job Failure Report

**Jobs reviewed:** 7563, 7617, 7635, 7639  
**Host:** `chimay01` (all runs)  
**Script:** `scripts/calculco/phase8_p0_model_ready.sh`  
**Current package status:** incomplete — `processed/phase8_p0/model_ready/` does not exist  
**Current leftover:** only `subgraph_nodes.parquet` (~142 MB); edges/candidates/index missing  

---

## Executive summary

There are **two distinct failure modes**, not one repeated crash:

| Mode | Jobs | Symptom | Root cause |
|------|------|---------|------------|
| **A. DuckDB OOM in `subgraph_edges`** | 7563, 7617, 7639 | Chain dies after training/graph; subgraphs exit 1 | Edge join exceeds ~10–12 GB DuckDB limit |
| **B. Immediate SIGBUS on training** | 7635 (and 7638) | Dies in ~10–18s at training start; exit 135; empty `.err` | Native process kill (signal 7); unrelated to subgraph logic |

Mode A is the real Step 9 blocker. Mode B is a transient early crash that only appears when the job restarts from training.

A third operational issue makes retries worse: **login-shell `export`s never reach OAR `-S` jobs**, so intended `PHASE8_P0_START_AT=subgraphs` never applied. Every job below logged `START_AT=training` (or had no START_AT field yet and behaved as a full chain).

---

## Shared configuration (all jobs)

| Setting | Value |
|---------|-------|
| Cores | 8 |
| Walltime request | 48h |
| `DUCKDB_THREADS` | 4 |
| `DUCKDB_MEMORY_LIMIT` | **10GB** |
| Scratch | `/tmp/rm-scratch/rm-job-<id>/tmp` |
| Feature rebuild | skipped (`auto`, features already present) |
| Token strategy | `rxnorm_or_atc` |

None of these jobs had a durable `phase8_p0_model_ready_job.env` at submit time.

---

## Job-by-job analysis

### Job 7563 — first OOM (nodes not finished)

| Field | Value |
|-------|-------|
| Started (UTC) | 2026-07-16T22:45:51Z |
| Duration | ~13 min |
| OAR state | `T` (terminated) |
| Knobs logged | No `SUBGRAPH_BATCHES` / join shards (earlier script revision) |

**Progress**

1. Preflight OK  
2. Training OK (4 tables)  
3. ATC3 sensitivity OK  
4. Preprocessing OK (~12.4M rows)  
5. Graph suitability OK  
6. Patient subgraphs **failed** — manifest `tables=1`

**Cause**

DuckDB out-of-memory while building patient subgraphs. With `tables=1`, failure happened during or right after the first subgraph table materialization (before a completed `subgraph_nodes` record was left in a later run’s shape). Same class of error as later jobs: spill/offload failed under the 10 GB cap.

**`.err`**

Only sklearn imputation warnings from preprocessing (empty platelet / hospital fields). Not the failure.

---

### Job 7617 — OOM after nodes completed (old edge join)

| Field | Value |
|-------|-------|
| Started (UTC) | 2026-07-17T11:09:24Z |
| Duration | ~18 min |
| Knobs logged | `SUBGRAPH_BATCHES=8` only |

**Progress**

Same successful path through training → ATC3 → preprocessing → graph, then:

```text
Wrote patient-subgraph manifest: status=failed, tables=2
```

**Cause (confirmed from prior manifest for this era)**

- `subgraph_nodes`: completed — **94,813,196** rows in 8 stay-hash batches  
- `subgraph_edges`: failed on **batch 0**  
- Reason:  
  `Out of Memory Error: failed to offload data block of size 256.0 KiB (12.2 GiB/12.2 GiB used).`

**Technical mechanism**

Edge construction still used a heavy join pattern: train-fit `graph_edges` (~391k rows) joined to a ~12M-row node batch, then self-joined on `subgraph_id` via string node IDs. Intermediate join state exceeded DuckDB’s configured memory (~10 GB request; reported use **12.2 GiB**).

**What this job still achieved**

Primary + ATC3 training tables, model-ready `cohort_stays`, preprocessing, graph edges, and the eICU readiness signal (RxNorm coverage-only; ATC3 externally evaluable) were successfully written.

---

### Job 7635 — SIGBUS, never reached subgraphs

| Field | Value |
|-------|-------|
| Started (UTC) | 2026-07-17T13:08:58Z |
| Duration | **~10 s** |
| Exit | **135** (= 128 + 7 → **SIGBUS**) |
| `.err` | **empty (0 bytes)** |

**Intended vs actual**

Login shell had:

```bash
export PHASE8_P0_START_AT=subgraphs
export SUBGRAPH_JOIN_SHARDS=8
...
```

Job log showed:

```text
PHASE8_P0_START_AT=training
```

OAR `-S` starts a clean environment. Shell exports were ignored. Script defaults applied. Job restarted from training instead of resuming at subgraphs.

**Cause**

Immediate native crash during `uv run python -m pipeline.build_training_table`. No Python traceback. Training parquet/manifests were **not** updated (mtime still from earlier successful run). Classic SIGBUS pattern on Calculco: mmap/NFS/native abort, not a handled DuckDB OOM.

**Why it matters**

This failure is **orthogonal** to the subgraph OOM. It only happens when the chain re-enters training. A true `START_AT=subgraphs` resume would skip it.

---

### Job 7639 — OOM on first *sharded* edge part (new code still too heavy)

| Field | Value |
|-------|-------|
| Started (UTC) | 2026-07-17T13:33:04Z |
| Duration | ~17 min |
| Knobs logged | `SUBGRAPH_BATCHES=8`, `SUBGRAPH_JOIN_SHARDS=8`, `SUBGRAPH_EDGE_THREADS=1`, `START_AT=training` |

**Progress**

Full successful rebuild again through training/ATC3/preprocessing/graph, then subgraphs failed with `tables=2`.

**Cause (current manifest, written by this job)**

```text
status: failed
reason: Out of Memory Error: failed to offload data block ... (12.2 GiB/12.2 GiB used)
```

| Table | Status | Detail |
|-------|--------|--------|
| `subgraph_nodes` | completed | 94,813,196 rows; 8 batches |
| `subgraph_edges` | failed | `encoded_relation_specific_join_shards`; **0/64** parts done |

Failure coordinates:

- `failed_node_batch_index`: **0**  
- `failed_join_shard_index`: **0**  
- `failed_batch_index`: **0**  
- `edge_part_count`: **64** (8 node batches × 8 join shards)

**What changed vs 7617**

The new encoded/sharded edge path **did run**. It is no longer the old string-key self-join over a full ~12M-node batch. But **even the first 1/64 slice** still exhausts ~12.2 GiB under `DUCKDB_MEMORY_LIMIT=10GB`. So sharding reduced work per query, but not enough for this hardware/memory envelope.

**`.err`**

Same sklearn warnings only.

---

## Comparative timeline

```text
7563  ~13m   train✓ … graph✓ → subgraphs fail (tables=1)     [early OOM]
7617  ~18m   train✓ … graph✓ → nodes✓ → edges fail batch0    [old join OOM]
7635  ~10s   train start → exit 135 SIGBUS                   [env + crash]
7638  ~18s   same as 7635 (related retry)                    [SIGBUS]
7639  ~17m   train✓ … graph✓ → nodes✓ → edges fail shard0/64 [new join OOM]
```

---

## Root-cause taxonomy

### 1. Primary blocker — subgraph edge memory (7563, 7617, 7639)

**What fails:** `pipeline.patient_subgraphs` materializing `subgraph_edges.parquet`  
**Where:** first edge batch/shard, after nodes succeed  
**DuckDB report:** cannot offload a 256 KiB block; **12.2 / 12.2 GiB** used  
**Config pressure:** `DUCKDB_MEMORY_LIMIT=10GB`, spill on node-local `/tmp`, 4 threads (edges forced to 1 thread in newer code)

**Scale context**

- ~95M subgraph nodes globally  
- ~12M nodes per stay-hash batch  
- With 8 join shards ≈ ~1.5M nodes per shard still too heavy for the current join plan at 10 GB  

**Code evolution**

| Era | Strategy | Outcome |
|-----|----------|---------|
| 7563 / 7617 | Stay-hash batches; heavy node/edge join | OOM on edges |
| 7639 | Encoded relation-specific join shards (64 parts) | Still OOM on part 0 |

So this is **not** “jobs mysteriously fail.” It is a reproducible memory ceiling on the edge stage.

### 2. Secondary failure — training SIGBUS (7635, 7638)

**What fails:** process killed with signal 7 before writing training outputs  
**Evidence:** exit 135, empty stderr, ~10–18s runtime, artifacts unchanged  
**Likely class:** transient native/mmap issue on compute node, not application logic  
**Mitigation:** resume at `subgraphs` so training is not re-entered

### 3. Operational amplifier — OAR env not propagated

**Effect:** every resume attempt still started at `training`  
**Evidence:** logs show `PHASE8_P0_START_AT=training`; no `phase8_p0_model_ready_job.env` present for these submits  
**Consequence:** wasted recomputation of already-good stages; exposed jobs to SIGBUS; delayed testing of higher `SUBGRAPH_JOIN_SHARDS`

A submit wrapper + job.env pattern was added later to fix this; these four jobs predate using it successfully.

---

## What is *not* the cause

- Preflight / missing harmonized inputs — all jobs passed preflight  
- Feature rebuild — correctly skipped  
- Sklearn platelet/hospital warnings — noise only  
- Graph suitability / train-only leakage gate — completed when reached  
- Training/ATC3/preprocessing logic — completed whenever the process stayed alive past the first few seconds  

---

## Current artifact state after these jobs

| Artifact family | Status |
|-----------------|--------|
| Phase 8 P0 features | Present (reused) |
| Training + ATC3 + cohort_stays + PCM | Present (last successful writes from 7617/7639 path) |
| Preprocessing | Present |
| Graph edges + suitability | Present |
| `subgraph_nodes` | Present (~95M rows) |
| `subgraph_edges` / candidates / index | **Missing** |
| Vocabularies / data dictionary / model-ready package | **Missing** |

Step 9 is blocked exclusively on finishing the subgraph edge (and subsequent) tables, then the package assembler.

---

## Conclusions

1. **7563, 7617, 7639** share one scientific/engineering cause: **DuckDB OOM in `subgraph_edges`**, with progressive evidence that even sharded encoded joins still do not fit in a 10 GB budget on the first shard.  
2. **7635** failed for a **different** reason: **SIGBUS during training restart**, made worse because **resume env vars never reached the job**.  
3. None of these jobs produced a completed model-ready package.  
4. Upstream model-ready pieces (features, training, ATC3, preprocessing, graph) are largely already on disk and usable; the remaining work is memory-safe edge materialization (and then package assembly), preferably via a true `subgraphs` resume with durable job.env and higher join shards / memory.

---

