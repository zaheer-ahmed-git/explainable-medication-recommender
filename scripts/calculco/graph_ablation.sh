#!/bin/bash
#OAR -n rm_graph_ablation
#OAR -l /nodes=1/core=8,walltime=08:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_graph_ablation_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_graph_ablation_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/graph_ablation.sh"
#
# Builds Milestone 8B graph-aware ablation artifacts from completed Milestone 6,
# Milestone 7, and Milestone 8 graph-readiness outputs. Row-level scores and
# models stay local under $DATASET_ROOT/processed/evaluation/milestone8b/;
# aggregate-only reports are written under $PROJECT_HOME/reports/.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

milestone8b_env_file="$script_dir/milestone8b_job.env"
if [[ -f "$milestone8b_env_file" ]]; then
  echo "=== milestone8b_job.env ==="
  # shellcheck disable=SC1090
  set -a
  source "$milestone8b_env_file"
  set +a
  cat "$milestone8b_env_file"
fi

mkdir -p "$script_dir/logs"

features_root="$DATASET_ROOT/processed/features"
training_root="$DATASET_ROOT/processed/training"
graph_root="$DATASET_ROOT/processed/graph/milestone8"
milestone7_evaluation_root="$DATASET_ROOT/processed/evaluation/milestone7"
evaluation_root="$DATASET_ROOT/processed/evaluation/milestone8b"

echo "=== graph_ablation preflight ==="
preflight_fail=0
check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "preflight ok: $label"
  else
    echo "preflight MISSING: $label ($path)" >&2
    preflight_fail=1
  fi
}

check_file "feature patient_stay_features" "$features_root/patient_stay_features.parquet"
check_file "training candidate_catalog" "$training_root/candidate_catalog.parquet"
check_file "training patient_condition_medication" "$training_root/patient_condition_medication.parquet"
check_file "graph edges" "$graph_root/graph_edges.parquet"
check_file "Milestone 7 baseline scores" "$milestone7_evaluation_root/baseline_scores.parquet"
check_file "feature manifest" "$PROJECT_HOME/reports/milestone6_feature_manifest.json"
check_file "training manifest" "$PROJECT_HOME/reports/training_table_manifest.json"
check_file "Milestone 7 final evaluation" "$PROJECT_HOME/reports/milestone7_baseline_evaluation.json"
check_file "Milestone 8 graph suitability" "$PROJECT_HOME/reports/milestone8_graph_suitability.json"

: "${MILESTONE8B_MODE:=development}"
if [[ "$MILESTONE8B_MODE" == "final" ]]; then
  check_file "Milestone 8B frozen selection" "$PROJECT_HOME/reports/milestone8b_frozen_selection.json"
fi

if (( preflight_fail != 0 )); then
  echo "Preflight failed; finish Milestone 6, Milestone 7 final, and Milestone 8 graph gate first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
echo "graph_root=$graph_root"
echo "milestone7_evaluation_root=$milestone7_evaluation_root"
echo "evaluation_root=$evaluation_root"

# Bound DuckDB work to the OAR allocation. Override by exporting before submit.
if [[ -z "${DUCKDB_THREADS:-}" ]]; then
  if [[ -n "${OAR_NODE_FILE:-}" && -f "${OAR_NODE_FILE}" ]]; then
    DUCKDB_THREADS="$(wc -l < "$OAR_NODE_FILE" | tr -d '[:space:]')"
  fi
  : "${DUCKDB_THREADS:=4}"
  export DUCKDB_THREADS
fi
if [[ -z "${DUCKDB_MEMORY_LIMIT:-}" ]]; then
  mem_gb=$(( DUCKDB_THREADS * 3 ))
  (( mem_gb < 6 )) && mem_gb=6
  export DUCKDB_MEMORY_LIMIT="${mem_gb}GB"
fi
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"

: "${MILESTONE8B_TOP_K:=1,3,5,10}"
: "${MILESTONE8B_FROZEN_SELECTION:=0}"
: "${MILESTONE8B_CONDITION_TOKENS:=}"

args=(
  --features-root "$features_root"
  --training-root "$training_root"
  --graph-root "$graph_root"
  --milestone7-evaluation-root "$milestone7_evaluation_root"
  --evaluation-root "$evaluation_root"
  --mode "$MILESTONE8B_MODE"
  --top-k "$MILESTONE8B_TOP_K"
)
if [[ "$MILESTONE8B_FROZEN_SELECTION" == "1" ]]; then
  args+=(--frozen-selection)
fi
if [[ -n "$MILESTONE8B_CONDITION_TOKENS" ]]; then
  args+=(--condition-token "$MILESTONE8B_CONDITION_TOKENS")
fi

echo "=== graph_ablation start ==="
if uv run python -m pipeline.graph_ablation "${args[@]}"; then
  ablation_rc=0
else
  ablation_rc=$?
fi
echo "=== graph_ablation done exit=$ablation_rc ==="

if (( ablation_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  milestone8b_graph_feature_manifest.json"
  echo "  milestone8b_ablation_evaluation.json"
  echo "  milestone8b_frozen_selection.json"
  echo "Local row-level scores and models stay ignored under $evaluation_root/"
fi

exit "$ablation_rc"
