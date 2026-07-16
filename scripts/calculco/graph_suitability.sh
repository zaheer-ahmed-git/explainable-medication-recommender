#!/bin/bash
#OAR -n rm_graph_suitability
#OAR -l /nodes=1/core=8,walltime=04:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/graph_suitability.sh"
#
# Builds Milestone 8 graph-readiness artifacts from completed Milestone 6
# artifacts. Concept-level graph edges stay local under
# $DATASET_ROOT/processed/graph/milestone8/; aggregate-only reports are written
# under $PROJECT_HOME/reports/.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

features_root="$DATASET_ROOT/processed/features"
training_root="$DATASET_ROOT/processed/training"
graph_root="$DATASET_ROOT/processed/graph/milestone8"

echo "=== graph_suitability preflight ==="
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

check_file "feature event_sequences" "$features_root/event_sequences.parquet"
check_file "training candidate_catalog" "$training_root/candidate_catalog.parquet"
check_file "training patient_condition_medication" "$training_root/patient_condition_medication.parquet"
check_file "feature manifest" "$PROJECT_HOME/reports/milestone6_feature_manifest.json"
check_file "training manifest" "$PROJECT_HOME/reports/training_table_manifest.json"
check_file "Milestone 7 frozen selection" "$PROJECT_HOME/reports/milestone7_frozen_selection.json"

if (( preflight_fail != 0 )); then
  echo "Preflight failed; finish Milestone 6 and freeze Milestone 7 selection first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
echo "graph_root=$graph_root"

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

echo "=== graph_suitability start ==="
if uv run python -m pipeline.graph_suitability \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-root "$graph_root"; then
  graph_rc=0
else
  graph_rc=$?
fi
echo "=== graph_suitability done exit=$graph_rc ==="

if (( graph_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  milestone8_graph_schema.json"
  echo "  milestone8_graph_suitability.json"
  echo "  milestone8_ablation_plan.json"
  echo "Local concept-level graph edges stay ignored under $graph_root/"
fi

exit "$graph_rc"
