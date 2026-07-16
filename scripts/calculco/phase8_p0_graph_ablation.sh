#!/bin/bash
#OAR -n rm_phase8_p0_graph_ablation
#OAR -l /nodes=1/core=8,walltime=08:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_ablation_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_ablation_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_graph_ablation.sh"
#
# Runs Milestone 8B graph-aware ablation on isolated Phase 8 P0 roots.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

phase8_job_env="$script_dir/phase8_p0_milestone8b_job.env"
if [[ -f "$phase8_job_env" ]]; then
  echo "=== phase8_p0_milestone8b_job.env ==="
  # shellcheck disable=SC1090
  set -a
  source "$phase8_job_env"
  set +a
  cat "$phase8_job_env"
fi

mkdir -p "$script_dir/logs"

phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
graph_root="${PHASE8_P0_GRAPH_ROOT:-$phase8_root/graph/milestone8}"
milestone7_evaluation_root="${PHASE8_P0_MILESTONE7_EVAL_ROOT:-$phase8_root/evaluation/milestone7}"
evaluation_root="${PHASE8_P0_MILESTONE8B_EVAL_ROOT:-$phase8_root/evaluation/milestone8b}"
feature_manifest="${PHASE8_P0_MILESTONE8B_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone8b_graph_feature_manifest.json}"
evaluation_report="${PHASE8_P0_MILESTONE8B_EVAL_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8b_ablation_evaluation.json}"
frozen_selection_report="${PHASE8_P0_MILESTONE8B_FROZEN_SELECTION:-$PROJECT_HOME/reports/phase8_p0_milestone8b_frozen_selection.json}"
milestone6_feature_manifest="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"
training_manifest="${PHASE8_P0_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json}"
milestone7_evaluation_report="${PHASE8_P0_MILESTONE7_EVAL_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone7_baseline_evaluation.json}"
milestone8_suitability_report="${PHASE8_P0_GRAPH_SUITABILITY_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json}"

echo "=== phase8_p0_graph_ablation preflight ==="
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
check_file "feature manifest" "$milestone6_feature_manifest"
check_file "training manifest" "$training_manifest"
check_file "Milestone 7 evaluation report" "$milestone7_evaluation_report"
check_file "Milestone 8 graph suitability" "$milestone8_suitability_report"

: "${MILESTONE8B_MODE:=development}"
if [[ "$MILESTONE8B_MODE" == "final" ]]; then
  check_file "Phase 8 P0 Milestone 8B frozen selection" "$frozen_selection_report"
fi

if (( preflight_fail != 0 )); then
  echo "Preflight failed; run phase8_p0_graph_suitability and milestone7 eval first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
echo "graph_root=$graph_root"
echo "milestone7_evaluation_root=$milestone7_evaluation_root"
echo "evaluation_root=$evaluation_root"

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
: "${MILESTONE8B_ALLOW_DEVELOPMENT_MILESTONE7_REFERENCE:=1}"

args=(
  --features-root "$features_root"
  --training-root "$training_root"
  --graph-root "$graph_root"
  --milestone7-evaluation-root "$milestone7_evaluation_root"
  --evaluation-root "$evaluation_root"
  --feature-manifest "$feature_manifest"
  --evaluation-report "$evaluation_report"
  --frozen-selection-report "$frozen_selection_report"
  --milestone6-feature-manifest "$milestone6_feature_manifest"
  --training-manifest "$training_manifest"
  --milestone7-evaluation-report "$milestone7_evaluation_report"
  --milestone8-suitability-report "$milestone8_suitability_report"
  --mode "$MILESTONE8B_MODE"
  --top-k "$MILESTONE8B_TOP_K"
)
if [[ "$MILESTONE8B_FROZEN_SELECTION" == "1" ]]; then
  args+=(--frozen-selection)
fi
if [[ "$MILESTONE8B_ALLOW_DEVELOPMENT_MILESTONE7_REFERENCE" == "1" ]]; then
  args+=(--allow-development-milestone7-reference)
fi
if [[ -n "$MILESTONE8B_CONDITION_TOKENS" ]]; then
  args+=(--condition-token "$MILESTONE8B_CONDITION_TOKENS")
fi

echo "=== phase8_p0_graph_ablation start ==="
if uv run python -m pipeline.graph_ablation "${args[@]}"; then
  ablation_rc=0
else
  ablation_rc=$?
fi
echo "=== phase8_p0_graph_ablation done exit=$ablation_rc ==="

if (( ablation_rc == 0 )); then
  echo "Review aggregate reports:"
  echo "  $feature_manifest"
  echo "  $evaluation_report"
  echo "  $frozen_selection_report"
  echo "Local row-level scores stay ignored under $evaluation_root/"
  echo "Run feature gate review on the login node:"
  echo "  uv run python -m pipeline.feature_gate_review \\"
  echo "    --phase8-evaluation-report $evaluation_report \\"
  echo "    --output $PROJECT_HOME/reports/phase8_p0_feature_gate_review.json"
fi

exit "$ablation_rc"
