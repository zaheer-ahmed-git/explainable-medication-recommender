#!/bin/bash
#OAR -n rm_phase8_p0_eval
#OAR -l /nodes=1/core=8,walltime=08:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_eval_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_eval_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_evaluate_baselines.sh"
#
# Runs Milestone 7 development (or final) baseline evaluation on isolated
# Phase 8 P0 model-ready artifacts. Does not overwrite canonical milestone7
# reports or evaluation roots.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

phase8_job_env="$script_dir/phase8_p0_milestone7_job.env"
if [[ -f "$phase8_job_env" ]]; then
  echo "=== phase8_p0_milestone7_job.env ==="
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
evaluation_root="${PHASE8_P0_EVALUATION_ROOT:-$phase8_root/evaluation/milestone7}"
feature_manifest="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"
training_manifest="${PHASE8_P0_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json}"
preprocessing_manifest="${PHASE8_P0_PREPROCESSING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_preprocessing_manifest.json}"
coverage_report="${PHASE8_P0_COVERAGE_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone7_coverage_report.json}"
evaluation_report="${PHASE8_P0_EVALUATION_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone7_baseline_evaluation.json}"

echo "=== phase8_p0_evaluate_baselines preflight ==="
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
check_file "preprocessing train_fitted_preprocessor" "$training_root/preprocessing/train_fitted_preprocessor.joblib"
check_file "feature manifest" "$feature_manifest"
check_file "training manifest" "$training_manifest"
check_file "preprocessing manifest" "$preprocessing_manifest"

if (( preflight_fail != 0 )); then
  echo "Preflight failed; run phase8_p0_model_ready.sh first." >&2
  exit 1
fi

echo "features_root=$features_root"
echo "training_root=$training_root"
echo "evaluation_root=$evaluation_root"
echo "coverage_report=$coverage_report"
echo "evaluation_report=$evaluation_report"

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
: "${PHASE8_P0_FEATURE_VERSION:=temporal-features-v2}"

args=(
  --features-root "$features_root"
  --training-root "$training_root"
  --evaluation-root "$evaluation_root"
  --coverage-report "$coverage_report"
  --evaluation-report "$evaluation_report"
  --training-manifest "$training_manifest"
  --mode "$MILESTONE7_MODE"
  --top-k "$MILESTONE7_TOP_K"
  --feature-version "$PHASE8_P0_FEATURE_VERSION"
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

echo "=== phase8_p0_evaluate_baselines start ==="
if uv run python -m pipeline.evaluate_baselines "${args[@]}"; then
  eval_rc=0
else
  eval_rc=$?
fi
echo "=== phase8_p0_evaluate_baselines done exit=$eval_rc ==="

if (( eval_rc == 0 )); then
  echo "Review aggregate reports:"
  echo "  $coverage_report"
  echo "  $evaluation_report"
  echo "Local row-level scores stay ignored under $evaluation_root/"
fi

exit "$eval_rc"
