#!/usr/bin/env bash
#OAR -n rm_phase8_p0_gnn_training
#OAR -l /nodes=1/gpu=1,walltime=48:00:00
#OAR -p gpudevice<>'-1'
# GPU training, development scoring, and explicitly confirmed one-shot final
# scoring for the relation-aware GNN and frozen-Transformer fusion workflow.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

job_env="$script_dir/phase8_p0_gnn_training_job.env"
if [[ -f "$job_env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$job_env"
  set +a
fi

: "${WORK_SCRATCH:?WORK_SCRATCH must be exported for GNN jobs}"
: "${GNN_MODE:=development}"
: "${GNN_STAGES:=train-gnn score-gnn train-fusion score-fusion}"
: "${GNN_SEED:=20260617}"
: "${GNN_TOP_K:=1,3,5,10}"
: "${GNN_FOLD_COUNT:=5}"
: "${GNN_SHARD_COUNT:=256}"
: "${GNN_DEVICE:=cuda}"
: "${DUCKDB_THREADS:=8}"
: "${DUCKDB_MEMORY_LIMIT:=24GB}"

case "$DUCKDB_TEMP_DIR" in
  "$WORK_SCRATCH"/*) ;;
  *)
    echo "GNN jobs require writable scratch under WORK_SCRATCH." >&2
    exit 1
    ;;
esac

if [[ "$GNN_MODE" == "final" ]]; then
  if [[ "$GNN_STAGES" != "score-gnn" && "$GNN_STAGES" != "score-fusion" ]]; then
    echo "Final mode permits exactly one one-shot scoring stage." >&2
    exit 2
  fi
  if [[ "${GNN_FINAL_SCORE_CONFIRM:-}" != "I_UNDERSTAND_ONE_SHOT" ]]; then
    echo "Set GNN_FINAL_SCORE_CONFIRM=I_UNDERSTAND_ONE_SHOT for final scoring." >&2
    exit 2
  fi
elif [[ "$GNN_MODE" != "development" ]]; then
  echo "GNN_MODE must be development or final." >&2
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

echo "=== Syncing the GNN/fusion runtime ==="
uv sync --group neural

common_args=(
  --mode "$GNN_MODE"
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
  --seed "$GNN_SEED"
  --top-k "$GNN_TOP_K"
  --fold-count "$GNN_FOLD_COUNT"
  --shard-count "$GNN_SHARD_COUNT"
  --device "$GNN_DEVICE"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)
if [[ -n "${GNN_BATCH_RANKING_GROUPS:-}" ]]; then
  common_args+=(--batch-ranking-groups "$GNN_BATCH_RANKING_GROUPS")
fi
if [[ -n "${GNN_MAX_EPOCHS:-}" ]]; then
  common_args+=(--max-epochs "$GNN_MAX_EPOCHS")
fi
if [[ -n "${GNN_EARLY_STOPPING_PATIENCE:-}" ]]; then
  common_args+=(--early-stopping-patience "$GNN_EARLY_STOPPING_PATIENCE")
fi
if [[ -n "${GNN_LEARNING_RATE:-}" ]]; then
  common_args+=(--learning-rate "$GNN_LEARNING_RATE")
fi
if [[ "$GNN_MODE" == "final" ]]; then
  common_args+=(--frozen-selection)
fi

for stage in $GNN_STAGES; do
  echo "=== Phase 8 P0 GNN stage: $stage ($GNN_MODE) ==="
  uv run python -m pipeline.gnn_training "$stage" "${common_args[@]}"
done

echo "Review aggregate reports under $reports_root; restricted artifacts remain under $gnn_root."
