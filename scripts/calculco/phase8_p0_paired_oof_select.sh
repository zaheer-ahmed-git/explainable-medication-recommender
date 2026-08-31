#!/usr/bin/env bash
#OAR -n rm_phase8_p0_paired_select
#OAR -l /nodes=1/core=16,walltime=04:00:00
#OAR -p gpudevice='-1'
# CPU-only joint GNN-variant/alpha selection after both OOF arrays complete.

set -euo pipefail

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

: "${WORK_SCRATCH:?WORK_SCRATCH must be exported for paired OOF selection}"
: "${PAIRED_SEED:=20260617}"
: "${PAIRED_FOLD_COUNT:=5}"
: "${PAIRED_GNN_SHARD_COUNT:=256}"
: "${DUCKDB_THREADS:=16}"
: "${DUCKDB_MEMORY_LIMIT:=48GB}"

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

uv sync --group neural
uv run python -m pipeline.gnn_training select-paired-oof \
  --mode development \
  --gnn-root "$gnn_root" \
  --neural-root "$neural_root" \
  --graph-root "$graph_root" \
  --subgraphs-root "$subgraphs_root" \
  --features-root "$phase8_root/features" \
  --training-root "$phase8_root/training" \
  --graph-reference-scores "$phase8_root/evaluation/milestone8b/graph_ablation_scores.parquet" \
  --graph-reference-report "$reports_root/phase8_p0_milestone8b_ablation_evaluation.json" \
  --contract-lock "$reports_root/phase8_p0_training_contract_lock.json" \
  --subgraphs-manifest "$reports_root/phase8_p0_patient_subgraphs_manifest.json" \
  --neural-selection "$reports_root/phase8_p0_neural_training_selection.json" \
  --crossfit-graph-manifest "$reports_root/phase8_p0_gnn_crossfit_graph_manifest.json" \
  --gnn-selection "$reports_root/phase8_p0_gnn_training_selection.json" \
  --fusion-selection "$reports_root/phase8_p0_fusion_training_selection.json" \
  --seed "$PAIRED_SEED" \
  --fold-count "$PAIRED_FOLD_COUNT" \
  --shard-count "$PAIRED_GNN_SHARD_COUNT" \
  --device cpu \
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR" \
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
  --duckdb-threads "$DUCKDB_THREADS"

