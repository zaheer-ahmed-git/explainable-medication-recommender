#!/usr/bin/env bash
#OAR -n rm_phase8_p0_paired_oof
#OAR -l /nodes=1/gpu=1,walltime=48:00:00
#OAR -p (network_address='chimay33' or network_address='chimay34') and gpudevice<>'-1'
# Parallel worker for the versioned paired-OOF late-fusion protocol.

set -euo pipefail

task="${1:-}"
item="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

job_env="$script_dir/phase8_p0_paired_oof_job.env"
if [[ -f "$job_env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$job_env"
  set +a
fi

: "${WORK_SCRATCH:?WORK_SCRATCH must be exported for paired OOF jobs}"
: "${PAIRED_SEED:=20260617}"
: "${PAIRED_FOLD_COUNT:=5}"
: "${PAIRED_NEURAL_SHARD_COUNT:=8}"
: "${PAIRED_GNN_SHARD_COUNT:=256}"
: "${PAIRED_DEVICE:=cuda}"
: "${DUCKDB_THREADS:=8}"
: "${DUCKDB_MEMORY_LIMIT:=24GB}"

case "$DUCKDB_TEMP_DIR" in
  "$WORK_SCRATCH"/*|"${TMPDIR:-/not-set}"/*|/scratch/*|/tmp/*) ;;
  *)
    echo "Paired OOF jobs require a writable bounded scratch directory." >&2
    exit 1
    ;;
esac

if [[ "$PAIRED_FOLD_COUNT" != "5" ]]; then
  echo "The protected paired-OOF protocol is frozen at exactly five folds." >&2
  exit 2
fi

export DUCKDB_THREADS DUCKDB_MEMORY_LIMIT
export OMP_NUM_THREADS="$DUCKDB_THREADS"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONUNBUFFERED=1

phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
reports_root="${REPORTS_ROOT:-$PROJECT_HOME/reports}"
gnn_root="${PHASE8_P0_GNN_ROOT:-$phase8_root/gnn}"
neural_root="${PHASE8_P0_NEURAL_ROOT:-$phase8_root/neural}"
graph_root="${PHASE8_P0_GRAPH_ROOT:-$phase8_root/graph/milestone8}"
subgraphs_root="${PHASE8_P0_SUBGRAPHS_ROOT:-$graph_root/patient_subgraphs}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"

uv sync --group neural

# OAR can expose a GPU-labelled resource whose device is unavailable to the
# process.  Fail before touching a retry report or loading CUDA checkpoints so
# an allocation problem cannot masquerade as a model/checkpoint failure.
if [[ "$PAIRED_DEVICE" == cuda* ]]; then
  uv run python -c 'import sys, torch; ok = torch.cuda.is_available() and torch.cuda.device_count() > 0; print(f"CUDA preflight: available={ok} device_count={torch.cuda.device_count()}"); sys.exit(0 if ok else 3)'
fi

neural_args=(
  --mode development
  --features-root "$features_root"
  --training-root "$training_root"
  --neural-root "$neural_root"
  --gnn-root "$gnn_root"
  --reference-scores "$phase8_root/evaluation/gate_recovery/baseline_scores.parquet"
  --contract-lock "$reports_root/phase8_p0_training_contract_lock.json"
  --gate-selection "$reports_root/phase8_p0_gate_recovery_selection.json"
  --seed "$PAIRED_SEED"
  --fold-count "$PAIRED_FOLD_COUNT"
  --shard-count "$PAIRED_NEURAL_SHARD_COUNT"
  --device "$PAIRED_DEVICE"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)
if [[ -n "${PAIRED_TRANSFORMER_FIXED_EPOCHS:-}" ]]; then
  neural_args+=(--fixed-epochs "$PAIRED_TRANSFORMER_FIXED_EPOCHS")
fi

gnn_args=(
  --mode development
  --gnn-root "$gnn_root"
  --neural-root "$neural_root"
  --graph-root "$graph_root"
  --subgraphs-root "$subgraphs_root"
  --features-root "$features_root"
  --training-root "$training_root"
  --graph-reference-scores "$phase8_root/evaluation/milestone8b/graph_ablation_scores.parquet"
  --graph-reference-report "$reports_root/phase8_p0_milestone8b_ablation_evaluation.json"
  --contract-lock "$reports_root/phase8_p0_training_contract_lock.json"
  --subgraphs-manifest "$reports_root/phase8_p0_patient_subgraphs_manifest.json"
  --neural-selection "$reports_root/phase8_p0_neural_training_selection.json"
  --crossfit-graph-manifest "$reports_root/phase8_p0_gnn_crossfit_graph_manifest.json"
  --gnn-selection "$reports_root/phase8_p0_gnn_training_selection.json"
  --fusion-selection "$reports_root/phase8_p0_fusion_training_selection.json"
  --paired-frozen-gate "$reports_root/phase8_p0_paired_oof_frozen_gate.json"
  --seed "$PAIRED_SEED"
  --fold-count "$PAIRED_FOLD_COUNT"
  --shard-count "$PAIRED_GNN_SHARD_COUNT"
  --device "$PAIRED_DEVICE"
  --precision "${PAIRED_GNN_PRECISION:-bf16}"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)

case "$task" in
  transformer-fold)
    if [[ ! "$item" =~ ^[0-4]$ ]]; then
      echo "transformer-fold requires a held-out fold in 0..4." >&2
      exit 2
    fi
    uv run python -m pipeline.neural_training oof-fold \
      "${neural_args[@]}" --held-out-fold "$item"
    ;;
  gnn-variant)
    case "$item" in
      full|rank_only|no_message_passing|no_condition_medication|no_dense_lab_vital|no_lab_vital_intervention) ;;
      *)
        echo "gnn-variant received an unsupported ablation variant." >&2
        exit 2
        ;;
    esac
    uv run python -m pipeline.gnn_training materialize-gnn-oof \
      "${gnn_args[@]}" --ablation-variant "$item"
    ;;
  refit)
    uv run python -m pipeline.gnn_training refit-paired-gnn "${gnn_args[@]}"
    ;;
  score)
    if [[ "${PAIRED_GATE_SCORE_CONFIRM:-}" != "I_UNDERSTAND_ONE_SHOT" ]]; then
      echo "Set PAIRED_GATE_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT to score the gate." >&2
      exit 2
    fi
    uv run python -m pipeline.gnn_training score-paired-late "${gnn_args[@]}"
    ;;
  *)
    echo "Usage: $0 {transformer-fold FOLD|gnn-variant VARIANT|refit|score}" >&2
    exit 2
    ;;
esac
