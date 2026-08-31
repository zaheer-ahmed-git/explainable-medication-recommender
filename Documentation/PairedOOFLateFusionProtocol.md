# Paired-OOF Late-Fusion Protocol

Protocol version: `phase8-p0-paired-oof-late-fusion-v2`

Status: implemented; the first protected OOF attempt produced four completed
Transformer folds and exposed recoverable Calculco GPU-allocation and GNN
walltime failures. Targeted recovery jobs `22470` (Transformer fold 1) and
`22471` (full GNN variant) were submitted on 2026-08-28; selection and the new
frozen-gate evaluation remain blocked. This protocol replaces the asymmetric late-fusion evidence in which
GNN logits were out-of-fold but Transformer logits came from a full-train
refit.

## Locked selection rules

- Use the existing deterministic five patient folds (`seed=20260617`) for both
  branches.
- For each Transformer fold, fit vocabularies, normalization, priors, graph
  summaries, and model weights on the other four folds only.
- Use the fold-excluded GNN graph for the matching Transformer task.
- Train each Transformer for the frozen full-model epoch count (currently four
  epochs); do not early-stop or calibrate on its held-out fold.
- Materialize raw OOF logits for all six pre-registered GNN variants from the
  existing fold checkpoints.
- Require an exact candidate, label, rank, and fold match between Transformer
  and GNN OOF tables.
- Jointly select `(GNN variant, alpha)` on paired train OOF only. The alpha grid
  is inclusive `0.000..0.250` in increments of `0.005`; exact metric ties keep
  the smaller alpha and then the earlier pre-registered GNN variant.
- Blend within-ranking-group z-scores as
  `(1-alpha) * Transformer + alpha * GNN`.
- Freeze the pair before opening a newly isolated gate. The gate is scored
  exactly once and must not have participated in model, variant, epoch, or
  alpha selection.
- Apply the unchanged promotion gate: NDCG@10 lift at least `+0.005`, with
  MRR@10 and Hit@10 drops no worse than `-0.01`, versus the frozen Transformer
  scored on the same candidates.
- Adaptive fusion is authorized only when the selected global OOF blend gains
  less than `+0.005` NDCG@10 over the best paired-OOF base model.

## Execution graph

```text
5 Transformer OOF fold jobs ─┐
                              ├─> CPU joint variant/alpha selection
6 GNN variant OOF jobs ──────┘              │
                                             v
                              reuse/refit selected full GNN
                                             │
                                             v
                              one-shot newly frozen gate score
```

The five Transformer tasks and six GNN materialization tasks are independent
OAR arrays. Each task owns a distinct protected output root, so concurrent
writes do not overlap. GNN folds are scored sequentially inside one
variant-specific task; variants may run concurrently subject to Calculco GPU
availability.

## Commands

Export `PROJECT_HOME`, `DATASET_ROOT`, and `WORK_SCRATCH` first.

```bash
# Submit both independent arrays (5 Transformer + 6 GNN tasks).
scripts/calculco/submit_phase8_p0_paired_oof.sh all-oof

# Targeted recovery; these commands do not repeat completed tasks.
scripts/calculco/submit_phase8_p0_paired_oof.sh transformer-fold 1
scripts/calculco/submit_phase8_p0_paired_oof.sh gnn-variant full

# After every array task has a completed aggregate report:
scripts/calculco/submit_phase8_p0_paired_oof.sh select

# Reuses the existing exact full refit when compatible; otherwise fits once.
scripts/calculco/submit_phase8_p0_paired_oof.sh refit

# Only after a separately reviewed frozen-gate manifest exists:
PAIRED_GATE_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT \
  scripts/calculco/submit_phase8_p0_paired_oof.sh score
```

The scoring command never creates or guesses a gate. It requires
`reports/phase8_p0_paired_oof_frozen_gate.json`, claims an exclusive one-shot
marker before predictions, and leaves an interrupted or failed attempt claimed.

## First protected attempt and recovery controls

The first paired attempt (`22100..22110`, submitted 2026-08-27) established the
following operational evidence:

- Transformer folds 0, 2, 3, and 4 completed in 53m49s--58m48s.
- Transformer fold 1 and five GNN variants landed on `chimay31` GPU slot 1,
  which OAR labelled as a GPU resource although CUDA was unavailable to the
  process. Those tasks failed before producing usable predictions.
- The full GNN materializer ran on a usable `chimay33` GPU but reached the
  12-hour walltime before its atomic Parquet output could be committed. Its
  temporary file is incomplete and is not a selection input.

The worker now reserves 48 hours, uses the proven `chimay33`/`chimay34` pool,
and performs a CUDA visibility check before touching retry reports or loading
checkpoints. Python device resolution also fails explicitly when CUDA was
requested but is unavailable. The submission helper accepts one fold or one
variant so recovery can proceed without overwriting completed evidence.
The remaining five GNN variants will be released only after the full-variant
recovery establishes a successful protected runtime.
Both recovery jobs started on `chimay34` and passed the new CUDA preflight
(`available=True`, one allocation-visible device per job).

### Serial GNN variant chain (recommended)

After the first successful `full` materializer, run the remaining variants one
at a time with automatic hand-off:

```bash
export PROJECT_HOME DATASET_ROOT WORK_SCRATCH
tmux new -s paired-gnn-chain
PAIRED_GNN_CHAIN_MAIL=you@example.com \
  scripts/calculco/watch_paired_gnn_oof_chain.sh
```

The watcher polls OAR every two minutes by default, mails on every job end, stops
on failure, and verifies each aggregate report says `status=completed` before
submitting the next variant. Optional CPU selection after the last variant:

```bash
PAIRED_GNN_CHAIN_ON_COMPLETE='scripts/calculco/submit_phase8_p0_paired_oof.sh select'
```

Built-in `oarsub --notify` may be unreliable on Calculco; prefer this watcher.
Test mail once with `scripts/calculco/send_oar_mail.sh`.

## Frozen gate manifest contract

The reviewed aggregate JSON must contain at least:

```json
{
  "schema_version": "phase8-p0-frozen-late-fusion-gate-v1",
  "protocol_version": "phase8-p0-paired-oof-late-fusion-v2",
  "status": "frozen",
  "gate_id": "reviewed-new-gate-id",
  "source": "mimiciv",
  "split": "reviewed_new_gate_split",
  "frozen_at": "ISO-8601 timestamp after paired OOF selection",
  "selection_completed_before_gate_opened": true,
  "one_shot_scoring_authorized": true,
  "used_for_model_selection": false,
  "used_for_gnn_variant_selection": false,
  "used_for_alpha_selection": false,
  "previously_scored_by_hybrid": false,
  "patient_overlap_with_train_count": 0,
  "paired_oof_selection_sha256": "reviewed digest",
  "gnn_cache_manifest_sha256": "reviewed digest",
  "transformer_cache_manifest_sha256": "reviewed digest"
}
```

Gate construction must also materialize matching GNN and frozen-Transformer
cache partitions for the named split. Defining that cohort/split is a research
decision outside this implementation and must be reviewed before data work.

### Gate-readiness evidence before selection

Aggregate-only review confirms that the existing MIMIC test partition has
3,423 patients, 65,062 ranking groups, and zero patient overlap with MIMIC
train. Matching GNN and frozen-Transformer caches already include `test`, and
the prior hybrid was scored on validation rather than test. However, the
frozen Transformer was previously scored on MIMIC test, so this split is not a
wholly unseen benchmark for the broader research program. It is the lowest-cost
operational gate only with that limitation disclosed. A genuinely unseen new
patient or temporal holdout has stronger evidence isolation but requires a new
reviewed cohort definition and cache materialization.

No frozen-gate manifest may be authored yet: `frozen_at` must follow paired-OOF
selection and its selection hash must match the completed artifact.

## Measured runtime basis and planning estimate

Completed Calculco jobs provide the following wall-clock basis:

| Prior job boundary | Actual wall time |
|---|---:|
| Transformer development prepare/train/score | 1h 50m 22s |
| GNN six-variant × five-fold sweep | 759.86 GPU-hours total |
| One successful GNN fold, median | 26h 28m 28s |
| GNN fold range | 13h 32m 53s–37h 18m 42s |
| Selected GNN full refit | 5h 15m 27s |
| Standalone GNN validation score | 40m 15s |
| Fusion training | 6h 22m 30s |
| Fusion validation score | 15m 30s |
| Paired Transformer OOF folds (4 successful) | 53m49s--58m48s each |
| First full GNN OOF materialization | exceeded 12h; killed at walltime |

Planning estimates (exclusive of queue wait and new-gate construction):

| New boundary | Estimated elapsed when parallel | Estimated compute |
|---|---:|---:|
| Five fold-isolated Transformer jobs | about 1h when five GPUs are available | about 4.7 GPU-hours measured/projected |
| Six GNN-variant OOF materializers | full variant exceeds 12h; reserve 48h per task | replace estimate after first completion |
| CPU joint selection (6 × 51 pairs) | about 1–3h | 16 CPU cores |
| Selected full GNN | minutes if exact refit reuses; otherwise about 5h15m | 0 or about 5.25 GPU-hours |
| One-shot gate score | about 15–45m | under 1 GPU-hour |

The earlier 6--10 hour GNN-materialization estimate was falsified by the first
full-variant job and must not be used for scheduling. The current operational
bound is a 48-hour reservation per variant, with parallel elapsed time governed
by usable GPU availability. This still reuses the 30 trained checkpoints and
avoids the 759.86 GPU-hour training sweep, but no revised compute saving is
claimed until one materializer completes.

## Safety and interpretation

All OOF and gate score tables remain restricted under protected storage. Public
reports contain aggregate metrics and artifact hashes only. Labels represent
observed historical prescriptions, not optimal treatment, and promotion is an
offline ranking decision for research and clinician review—not clinical advice.
