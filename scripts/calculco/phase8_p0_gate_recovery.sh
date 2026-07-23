#!/usr/bin/env bash
#OAR -n rm_phase8_p0_gate_recovery
#OAR -l /nodes=1/core=16,walltime=48:00:00
#OAR -p gpudevice='-1'
# Run the Phase 8 P0 contract audit and rank-aware structured recovery on CPU.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

job_env="$script_dir/phase8_p0_gate_recovery_job.env"
if [[ -f "$job_env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$job_env"
  set +a
fi

: "${GATE_RECOVERY_MODE:=development}"
: "${GATE_RECOVERY_TOP_K:=1,3,5,10}"
: "${GATE_RECOVERY_FOLD_COUNT:=3}"
: "${GATE_RECOVERY_SEED:=20260617}"
: "${DUCKDB_THREADS:=16}"
: "${DUCKDB_MEMORY_LIMIT:=48GB}"

export DUCKDB_THREADS DUCKDB_MEMORY_LIMIT
export OMP_NUM_THREADS="$DUCKDB_THREADS"

phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
graph_root="${PHASE8_P0_GRAPH_ROOT:-$phase8_root/graph/milestone8}"
reference_root="${PHASE8_P0_MILESTONE8B_EVAL_ROOT:-$phase8_root/evaluation/milestone8b}"
evaluation_root="${PHASE8_P0_GATE_RECOVERY_ROOT:-$phase8_root/evaluation/gate_recovery}"
contract_lock="${PHASE8_P0_CONTRACT_LOCK:-$PROJECT_HOME/reports/phase8_p0_training_contract_lock.json}"
contract_audit="${PHASE8_P0_CONTRACT_AUDIT:-$PROJECT_HOME/reports/phase8_p0_training_contract_audit_latest.json}"
selection_report="${PHASE8_P0_GATE_SELECTION:-$PROJECT_HOME/reports/phase8_p0_gate_recovery_selection.json}"
evaluation_report="${PHASE8_P0_GATE_EVALUATION:-$PROJECT_HOME/reports/phase8_p0_gate_recovery_evaluation.json}"
if [[ "$GATE_RECOVERY_MODE" == "final" ]]; then
  evaluation_report="${PHASE8_P0_GATE_FINAL_EVALUATION:-$PROJECT_HOME/reports/phase8_p0_gate_recovery_final_evaluation.json}"
fi

mkdir -p "$script_dir/logs"

contract_args=(
  --dataset-root "$DATASET_ROOT"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)
if [[ -f "$contract_lock" ]]; then
  contract_args+=(--expected-lock "$contract_lock" --output "$contract_audit")
else
  contract_args+=(--output "$contract_lock")
fi

echo "=== Phase 8 P0 training contract audit ==="
uv run python -m pipeline.training_contract "${contract_args[@]}"

recovery_args=(
  --mode "$GATE_RECOVERY_MODE"
  --features-root "$features_root"
  --training-root "$training_root"
  --graph-root "$graph_root"
  --reference-scores "$reference_root/_scores_reference.parquet"
  --reference-selection "$PROJECT_HOME/reports/phase8_p0_milestone8b_frozen_selection.json"
  --contract-lock "$contract_lock"
  --feature-manifest "$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json"
  --evaluation-root "$evaluation_root"
  --evaluation-report "$evaluation_report"
  --selection-report "$selection_report"
  --top-k "$GATE_RECOVERY_TOP_K"
  --seed "$GATE_RECOVERY_SEED"
  --fold-count "$GATE_RECOVERY_FOLD_COUNT"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)
if [[ "$GATE_RECOVERY_MODE" == "final" ]]; then
  recovery_args+=(--frozen-selection)
fi

echo "=== Phase 8 P0 gate recovery: $GATE_RECOVERY_MODE ==="
uv run python -m pipeline.gate_recovery "${recovery_args[@]}"

echo "Review aggregate reports only:"
echo "  $evaluation_report"
echo "  $selection_report"
echo "Patient-level scores and fitted artifacts remain under $evaluation_root"
