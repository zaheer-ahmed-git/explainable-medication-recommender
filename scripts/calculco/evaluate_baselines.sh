#!/bin/bash
#OAR -n rm_evaluate_baselines
#OAR -l /nodes=1/core=8,walltime=08:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_eval_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_eval_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/evaluate_baselines.sh"
#
# Builds Milestone 7 coverage reports and transparent baseline metrics
# (non-learned and learned) from completed Milestone 6 artifacts. Row-level scores are local
# ignored artifacts under $DATASET_ROOT/processed/evaluation/milestone7/;
# aggregate-only reports are written under $PROJECT_HOME/reports/.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

milestone7_env_file="$script_dir/milestone7_job.env"
if [[ -f "$milestone7_env_file" ]]; then
  echo "=== milestone7_job.env ==="
  # shellcheck disable=SC1090
  set -a
  source "$milestone7_env_file"
  set +a
  cat "$milestone7_env_file"
fi

mkdir -p "$script_dir/logs"

features_root="$DATASET_ROOT/processed/features"
training_root="$DATASET_ROOT/processed/training"
evaluation_root="$DATASET_ROOT/processed/evaluation/milestone7"

echo "=== evaluate_baselines preflight ==="
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
check_file "feature manifest" "$PROJECT_HOME/reports/milestone6_feature_manifest.json"
check_file "training manifest" "$PROJECT_HOME/reports/training_table_manifest.json"

if (( preflight_fail != 0 )); then
  echo "Preflight failed; run features.sh and build_training_table.sh first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
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

: "${MILESTONE7_MODE:=development}"
: "${MILESTONE7_TOP_K:=1,3,5,10}"
: "${MILESTONE7_FROZEN_SELECTION:=0}"
: "${MILESTONE7_CONDITION_TOKENS:=}"

: "${MILESTONE7_BASELINES:=random,global_popularity,condition_popularity,linear,xgboost}"

args=(
  --features-root "$features_root"
  --training-root "$training_root"
  --evaluation-root "$evaluation_root"
  --mode "$MILESTONE7_MODE"
  --top-k "$MILESTONE7_TOP_K"
)
if [[ "$MILESTONE7_FROZEN_SELECTION" == "1" ]]; then
  args+=(--frozen-selection)
fi
if [[ -n "$MILESTONE7_CONDITION_TOKENS" ]]; then
  args+=(--condition-token "$MILESTONE7_CONDITION_TOKENS")
fi
IFS=',' read -r -a selected_baselines <<< "$MILESTONE7_BASELINES"
for baseline in "${selected_baselines[@]}"; do
  trimmed="${baseline#"${baseline%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" ]]; then
    args+=(--baseline "$trimmed")
  fi
done

echo "=== evaluate_baselines start ==="
if uv run python -m pipeline.evaluate_baselines "${args[@]}"; then
  eval_rc=0
else
  eval_rc=$?
fi
echo "=== evaluate_baselines done exit=$eval_rc ==="

if (( eval_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  milestone7_coverage_report.json"
  echo "  milestone7_baseline_evaluation.json"
  echo "Local row-level scores stay ignored under $evaluation_root/"
fi

exit "$eval_rc"
