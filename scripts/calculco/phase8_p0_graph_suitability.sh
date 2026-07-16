#!/bin/bash
#OAR -n rm_phase8_p0_graph
#OAR -l /nodes=1/core=8,walltime=04:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_graph_suitability.sh"
#
# Builds Milestone 8 graph-readiness artifacts on isolated Phase 8 P0 roots.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
graph_root="${PHASE8_P0_GRAPH_ROOT:-$phase8_root/graph/milestone8}"
feature_manifest="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"
training_manifest="${PHASE8_P0_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json}"
schema_report="${PHASE8_P0_GRAPH_SCHEMA_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8_graph_schema.json}"
suitability_report="${PHASE8_P0_GRAPH_SUITABILITY_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json}"
ablation_plan="${PHASE8_P0_GRAPH_ABLATION_PLAN:-$PROJECT_HOME/reports/phase8_p0_milestone8_ablation_plan.json}"

echo "=== phase8_p0_graph_suitability preflight ==="
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
check_file "feature manifest" "$feature_manifest"
check_file "training manifest" "$training_manifest"

if (( preflight_fail != 0 )); then
  echo "Preflight failed; run phase8_p0_model_ready.sh first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
echo "graph_root=$graph_root"
echo "schema_report=$schema_report"
echo "suitability_report=$suitability_report"
echo "ablation_plan=$ablation_plan"

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

echo "=== phase8_p0_graph_suitability start ==="
if uv run python -m pipeline.graph_suitability \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-root "$graph_root" \
  --graph-schema-report "$schema_report" \
  --suitability-report "$suitability_report" \
  --ablation-plan "$ablation_plan"; then
  graph_rc=0
else
  graph_rc=$?
fi
echo "=== phase8_p0_graph_suitability done exit=$graph_rc ==="

if (( graph_rc == 0 )); then
  echo "Review aggregate reports:"
  echo "  $schema_report"
  echo "  $suitability_report"
  echo "  $ablation_plan"
  echo "Local graph edges stay ignored under $graph_root/"
fi

exit "$graph_rc"
