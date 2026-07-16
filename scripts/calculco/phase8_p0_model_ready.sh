#!/bin/bash
#OAR -n rm_phase8_p0_model_ready
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_model_ready_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_model_ready_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_model_ready.sh"
#
# Chains Phase 8 P0 features, training-table construction, and train-only
# preprocessing into isolated model-ready roots under
# $DATASET_ROOT/processed/phase8_p0.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

harmonized_root="${HARMONIZED_ROOT_OVERRIDE:-$DATASET_ROOT/processed/harmonized}"
phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
preprocessing_root="${PHASE8_P0_PREPROCESSING_ROOT:-$training_root/preprocessing}"
feature_manifest="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"
training_manifest="${PHASE8_P0_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json}"
preprocessing_manifest="${PHASE8_P0_PREPROCESSING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_preprocessing_manifest.json}"

echo "=== phase8_p0_model_ready preflight ==="
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
for tbl in cohort_stays demographics conditions medications labs vitals allergies interventions temporal_events; do
  check_file "$tbl" "$harmonized_root/$tbl.parquet"
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; run harmonization before Phase 8 P0 model-ready artifacts." >&2
  exit 1
fi

if [[ -z "${DUCKDB_THREADS:-}" ]]; then
  if [[ -n "${OAR_NODE_FILE:-}" && -f "${OAR_NODE_FILE}" ]]; then
    DUCKDB_THREADS="$(wc -l < "$OAR_NODE_FILE" | tr -d '[:space:]')"
    (( DUCKDB_THREADS > 4 )) && DUCKDB_THREADS=4
  fi
  : "${DUCKDB_THREADS:=4}"
  export DUCKDB_THREADS
fi
if [[ -z "${DUCKDB_MEMORY_LIMIT:-}" ]]; then
  export DUCKDB_MEMORY_LIMIT="10GB"
fi
: "${STAY_FEATURE_BATCHES:=8}"
: "${EVENT_SEQUENCE_BATCHES:=8}"
: "${CONDITION_FEATURE_TOP_N:=40}"
: "${TREND_MIN_EVENTS:=2}"
: "${CANDIDATE_TOKEN_STRATEGY:=rxnorm_or_atc}"
: "${PHASE8_P0_RUN_FEATURES:=auto}"
export STAY_FEATURE_BATCHES EVENT_SEQUENCE_BATCHES CONDITION_FEATURE_TOP_N
export TREND_MIN_EVENTS CANDIDATE_TOKEN_STRATEGY

echo "harmonized_root=$harmonized_root"
echo "features_root=$features_root"
echo "training_root=$training_root"
echo "preprocessing_root=$preprocessing_root"
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"
echo "PHASE8_P0_RUN_FEATURES=$PHASE8_P0_RUN_FEATURES"
echo "CANDIDATE_TOKEN_STRATEGY=$CANDIDATE_TOKEN_STRATEGY"

run_features=0
if [[ "$PHASE8_P0_RUN_FEATURES" == "1" ]]; then
  run_features=1
elif [[ "$PHASE8_P0_RUN_FEATURES" == "auto" && ! -f "$features_root/patient_stay_features.parquet" ]]; then
  run_features=1
fi

if (( run_features == 1 )); then
  echo "=== phase8_p0_features start ==="
  if uv run python -m pipeline.features \
    --feature-set phase8_p0 \
    --harmonized-root "$harmonized_root" \
    --features-root "$features_root" \
    --manifest "$feature_manifest" \
    --condition-feature-top-n "$CONDITION_FEATURE_TOP_N" \
    --trend-min-events "$TREND_MIN_EVENTS" \
    --stay-feature-batches "$STAY_FEATURE_BATCHES" \
    --event-sequence-batches "$EVENT_SEQUENCE_BATCHES"; then
    features_rc=0
  else
    features_rc=$?
  fi
  echo "=== phase8_p0_features done exit=$features_rc ==="
  if (( features_rc != 0 )); then
    exit "$features_rc"
  fi
else
  echo "Skipping Phase 8 P0 feature rebuild; set PHASE8_P0_RUN_FEATURES=1 to force."
fi

echo "=== phase8_p0_training_table start ==="
if uv run python -m pipeline.build_training_table \
  --harmonized-root "$harmonized_root" \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --manifest "$training_manifest" \
  --candidate-token-strategy "$CANDIDATE_TOKEN_STRATEGY"; then
  training_rc=0
else
  training_rc=$?
fi
echo "=== phase8_p0_training_table done exit=$training_rc ==="
if (( training_rc != 0 )); then
  exit "$training_rc"
fi

echo "=== phase8_p0_preprocessing start ==="
if uv run python -m pipeline.preprocessing \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --preprocessing-root "$preprocessing_root" \
  --manifest "$preprocessing_manifest"; then
  preprocessing_rc=0
else
  preprocessing_rc=$?
fi
echo "=== phase8_p0_preprocessing done exit=$preprocessing_rc ==="

if (( preprocessing_rc == 0 )); then
  echo "Review aggregate reports:"
  echo "  $feature_manifest"
  echo "  $training_manifest"
  echo "  $preprocessing_manifest"
  echo "Local patient-level artifacts stay ignored under $phase8_root/"
fi

exit "$preprocessing_rc"
