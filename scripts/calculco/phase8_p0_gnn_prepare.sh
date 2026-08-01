#!/usr/bin/env bash
#OAR -n rm_phase8_p0_gnn_prepare
# Request enough cores to land on 512 GB CPU nodes. DuckDB memory is bounded
# separately; keep threads low so hash-join fan-out stays spillable.
#OAR -l /nodes=1/core=24,walltime=48:00:00
#OAR -p gpudevice='-1'
# CPU preparation for fold-excluded GNN graph caches and frozen Transformer
# representations. Submit with submit_phase8_p0_gnn_prepare.sh only after a
# storage-capacity review.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

job_env="$script_dir/phase8_p0_gnn_prepare_job.env"
if [[ -f "$job_env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$job_env"
  set +a
fi

: "${WORK_SCRATCH:?WORK_SCRATCH must be exported for GNN preparation}"
: "${GNN_CROSSFIT_MIN_FREE_GIB:?Set a capacity-reviewed free-space threshold}"
: "${GNN_MODE:=development}"
: "${GNN_SEED:=20260617}"
: "${GNN_TOP_K:=1,3,5,10}"
: "${GNN_FOLD_COUNT:=5}"
: "${GNN_SHARD_COUNT:=256}"
: "${GNN_DEVICE:=cpu}"
# Cross-fit rebuilds scan subgraph_edges (~9.5GB compressed) with per-fold joins.
# Prefer fewer threads and a high DuckDB ceiling on 512 GB nodes; spill to
# WORK_SCRATCH rather than pinning a full MATERIALIZED edge join in RAM.
: "${DUCKDB_THREADS:=4}"
: "${DUCKDB_MEMORY_LIMIT:=128GB}"
: "${DUCKDB_MAX_TEMP_DIR_SIZE:=400GB}"

if [[ "$GNN_MODE" != "development" ]]; then
  echo "GNN preparation is a development-stage cache build." >&2
  exit 2
fi
case "$DUCKDB_TEMP_DIR" in
  "$WORK_SCRATCH"/*) ;;
  *)
    echo "GNN preparation requires writable scratch under WORK_SCRATCH." >&2
    exit 1
    ;;
esac

export GNN_CROSSFIT_MIN_FREE_GIB DUCKDB_THREADS DUCKDB_MEMORY_LIMIT
export DUCKDB_MAX_TEMP_DIR_SIZE
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

echo "=== Syncing the frozen-Transformer/GNN runtime ==="
uv sync --group neural

echo "=== Phase 8 P0 GNN prepare (CPU, capacity-gated) ==="
uv run python -m pipeline.gnn_training prepare \
  --mode "$GNN_MODE" \
  --gnn-root "$gnn_root" \
  --neural-root "$neural_root" \
  --graph-root "$graph_root" \
  --subgraphs-root "$subgraphs_root" \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-reference-scores "$phase8_root/evaluation/milestone8b/graph_ablation_scores.parquet" \
  --graph-reference-report "$reports_root/phase8_p0_milestone8b_ablation_evaluation.json" \
  --contract-lock "$reports_root/phase8_p0_training_contract_lock.json" \
  --subgraphs-manifest "$reports_root/phase8_p0_patient_subgraphs_manifest.json" \
  --neural-selection "$reports_root/phase8_p0_neural_training_selection.json" \
  --crossfit-graph-manifest "$reports_root/phase8_p0_gnn_crossfit_graph_manifest.json" \
  --seed "$GNN_SEED" \
  --top-k "$GNN_TOP_K" \
  --fold-count "$GNN_FOLD_COUNT" \
  --shard-count "$GNN_SHARD_COUNT" \
  --device "$GNN_DEVICE" \
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR" \
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
  --duckdb-max-temp-directory-size "$DUCKDB_MAX_TEMP_DIR_SIZE" \
  --duckdb-threads "$DUCKDB_THREADS"

echo "Review aggregate status only:"
echo "  $reports_root/phase8_p0_gnn_prepare_manifest.json"
echo "  $reports_root/phase8_p0_gnn_crossfit_graph_manifest.json"
echo "Preparation must report status=completed before GPU training."
