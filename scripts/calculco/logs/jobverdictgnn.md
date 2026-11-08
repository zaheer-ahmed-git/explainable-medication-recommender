Not bound to fail the same way as 10962 — and right now it looks healthier. Completion is still not guaranteed.

Current status (job 12214)
State
Running on chimay31 (GPU)
Runtime
~6h09m / 48h
Stage
Still train-gnn (no finish/fail line)
Resources
48 cores + 1 GPU (previous runs were 8 cores + 1 GPU)
New setting
GNN_MIXED_PRECISION=1
Logs are quiet after a clean start. That is normal for this stage.

Why this is different from the predecessors
Job	Failure
8825
Multi-file edge shard layout
10962
Hard crash on non-finite grads in clip_grad_norm_ (~6.5h)
12214
Past both so far; training code now uses AMP + error_if_nonfinite=False and retries overflowed batches instead of dying on the first NaN clip
The report file still shows the old 10962 failure — it is stale until this job finishes.

Caveats
No checkpoints/ yet after ~6h, so the first fold/variant has not finished writing artifacts.
AMP can still fail later if a batch stays non-finite after the max loss-scale retries.
First real success signal is either a checkpoint appearing, or
GNN train-gnn finished: status=completed.
Bottom line
It is working and not doomed to repeat 10962’s exact crash. It may still fail for a related numerical reason, or succeed if AMP backoff holds. Keep watching:

oarstat -j 12214
tail scripts/calculco/logs/rm_phase8_p0_gnn_training_12214.out
ls /nfs/data/protected/lisic/zahmed/ResearchModule/Dataset/processed/phase8_p0/gnn/checkpoints 2>/dev/null
That progress check finished cleanly and matches what we already saw for 12214:

Job scratch is almost empty (only uv cache/locks) — no checkpoint spill there
No .pt / checkpoint files under gnn/ yet
Mixed precision is wired on (GNN_MIXED_PRECISION=1), with AMP overflow retries in runtime.py
So the job is still in early/mid train-gnn without written fold artifacts.

**Verdict: still running, but not healthy — it is under GPU memory pressure and looks at risk.**

### What I see now (~8h05m)
| | |
|--|--|
| State | Still `Running` on `chimay31` |
| Stage | Still `train-gnn` (stdout has not advanced) |
| Checkpoints | None yet (stale report is still the old 10962 failure) |

### New warning (important)
`stderr` now has a **CUDA OOM** at ~01:51:

```text
memory allocation failed with OOM on device 0
trying to allocate ~5.94 GB
free: ~3.55 GB
total: ~42.4 GB
```

So the GPU is mostly full, and a large allocation failed. The job has not exited yet, but this is a bad sign.

### Compared with predecessors
| Job | Outcome |
|-----|---------|
| 8825 | Shard-layout crash |
| 10962 | Non-finite grads ~6.5h |
| **12214** | Past those so far, but now hitting **GPU OOM** around 8h |

### Bottom line
Not “working fine.” It has survived longer than 10962, but the CUDA OOM means it may hang, retry painfully, or fail soon unless memory use drops (smaller batches / fewer graph nodes in a batch).

Watch for either:
```bash
# failure
GNN train-gnn finished: status=failed
# or recovery/progress
ls .../gnn/checkpoints
tail scripts/calculco/logs/rm_phase8_p0_gnn_training_12214.err
```

That artifact scan finished and confirms the earlier verdict for **12214**:

- Training report on disk is still the **old 10962** failure (non-finite grads) — not updated by this job
- **No** `checkpoints/` or `predictions/`
- **No** new files under `gnn/` since this job started
- Job scratch is still essentially empty (~6 KB)

So after ~8h it still hadn’t written fold artifacts; the CUDA OOM in stderr remains the main live warning.

**Verdict: still alive, but looking poor — likely stalled or barely progressing after GPU OOM.**

### Snapshot (~8h49m)
| Signal | Finding |
|--------|---------|
| OAR state | Still `Running` on `chimay31` |
| Stdout | Unchanged — still only `train-gnn` start |
| Stderr | One CUDA OOM at **01:51**; **no new lines since** |
| Checkpoints | Still **none** |
| Training report | Still **stale** (old 10962 failure) |

### Interpretation
- It has **not** crashed like 10962 yet.
- The OOM (~6 GB alloc, ~3.5 GB free) happened ~1 h ago, then silence.
- After nearly **9 h** with no checkpoint and no log movement, this does **not** look like healthy training. More likely: recovering slowly, stuck in a huge batch, or hanging after the allocator failure.

### Bottom line
**Not fine.** Not confirmed dead either — but odds of a clean full run (4 ablations × 5 folds) from this job are low unless checkpoints start appearing soon.

If nothing changes in the next hour or two (no checkpoint, no new log line), cancel and restart with a much smaller batch (e.g. `GNN_BATCH_RANKING_GROUPS=1` or `2`) before burning more walltime.