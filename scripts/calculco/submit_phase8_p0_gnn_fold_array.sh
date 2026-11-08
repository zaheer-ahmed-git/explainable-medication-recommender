#!/usr/bin/env bash
# Submit one restartable OAR array task per pre-registered (variant, fold).

set -euo pipefail
umask 077

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_home="${PROJECT_HOME:-$(cd "$script_dir/../.." && pwd)}"
: "${WORK_SCRATCH:?Export WORK_SCRATCH before submitting the GNN fold array}"

case "$WORK_SCRATCH" in
  /path|/path/*)
    echo "WORK_SCRATCH looks like a documentation placeholder." >&2
    exit 2
    ;;
esac

job_env="$script_dir/phase8_p0_gnn_training_job.env"
{
  printf 'WORK_SCRATCH=%q\n' "$WORK_SCRATCH"
  printf 'GNN_MODE=development\n'
  printf 'GNN_STAGES=train-gnn-fold\n'
  printf 'GNN_SEED=%q\n' "${GNN_SEED:-20260617}"
  printf 'GNN_TOP_K=%q\n' "${GNN_TOP_K:-1,3,5,10}"
  printf 'GNN_FOLD_COUNT=%q\n' "${GNN_FOLD_COUNT:-5}"
  printf 'GNN_SHARD_COUNT=%q\n' "${GNN_SHARD_COUNT:-256}"
  printf 'GNN_BATCH_RANKING_GROUPS=%q\n' "${GNN_BATCH_RANKING_GROUPS:-32}"
  printf 'GNN_GRADIENT_ACCUMULATION_GROUPS=%q\n' "${GNN_GRADIENT_ACCUMULATION_GROUPS:-32}"
  printf 'GNN_MAX_EDGES_PER_BATCH=%q\n' "${GNN_MAX_EDGES_PER_BATCH:-100000}"
  printf 'GNN_MAX_NODES_PER_BATCH=%q\n' "${GNN_MAX_NODES_PER_BATCH:-8192}"
  printf 'GNN_PROGRESS_INTERVAL_BATCHES=%q\n' "${GNN_PROGRESS_INTERVAL_BATCHES:-100}"
  printf 'GNN_MAX_EPOCHS=%q\n' "${GNN_MAX_EPOCHS:-30}"
  printf 'GNN_EARLY_STOPPING_PATIENCE=%q\n' "${GNN_EARLY_STOPPING_PATIENCE:-3}"
  printf 'GNN_MIXED_PRECISION=%q\n' "${GNN_MIXED_PRECISION:-1}"
  printf 'GNN_PRECISION=%q\n' "${GNN_PRECISION:-bf16}"
  printf 'GNN_DEVICE=%q\n' "${GNN_DEVICE:-cuda}"
  printf 'DUCKDB_THREADS=%q\n' "${DUCKDB_THREADS:-8}"
  printf 'DUCKDB_MEMORY_LIMIT=%q\n' "${DUCKDB_MEMORY_LIMIT:-24GB}"
} >"$job_env"

parameter_file="$WORK_SCRATCH/phase8_p1_gnn_fold_array.params"
: >"$parameter_file"
fold_count="${GNN_FOLD_COUNT:-5}"
for variant in full rank_only no_message_passing no_condition_medication no_dense_lab_vital no_lab_vital_intervention; do
  for ((fold = 0; fold < fold_count; fold++)); do
    printf '%s %s\n' "$variant" "$fold" >>"$parameter_file"
  done
done

mkdir -p "$project_home/scripts/calculco/logs"
oarsub \
  --array-param-file "$parameter_file" \
  -O "$project_home/scripts/calculco/logs/rm_phase8_p1_gnn_fold_%jobid%.out" \
  -E "$project_home/scripts/calculco/logs/rm_phase8_p1_gnn_fold_%jobid%.err" \
  -S "$script_dir/phase8_p0_gnn_training.sh"
