#!/bin/bash
#OAR -n rm_phase8_p0_features
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_features_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_features_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_features.sh"
#
# Builds reviewed Phase 8 P0 temporal-features-v2 artifacts in isolated roots:
# $DATASET_ROOT/processed/phase8_p0/features and
# reports/phase8_p0_milestone6_feature_manifest.json.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

harmonized_root="${HARMONIZED_ROOT_OVERRIDE:-$DATASET_ROOT/processed/harmonized}"
phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
manifest_path="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"

echo "=== phase8_p0_features preflight ==="
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
for tbl in cohort_stays demographics conditions labs vitals allergies interventions temporal_events; do
  check_file "$tbl" "$harmonized_root/$tbl.parquet"
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; run harmonization before Phase 8 P0 features." >&2
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
export STAY_FEATURE_BATCHES EVENT_SEQUENCE_BATCHES CONDITION_FEATURE_TOP_N TREND_MIN_EVENTS

echo "harmonized_root=$harmonized_root"
echo "features_root=$features_root"
echo "manifest_path=$manifest_path"
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"
echo "STAY_FEATURE_BATCHES=$STAY_FEATURE_BATCHES"
echo "EVENT_SEQUENCE_BATCHES=$EVENT_SEQUENCE_BATCHES"
echo "CONDITION_FEATURE_TOP_N=$CONDITION_FEATURE_TOP_N"
echo "TREND_MIN_EVENTS=$TREND_MIN_EVENTS"

echo "=== phase8_p0_features start ==="
if uv run python -m pipeline.features \
  --feature-set phase8_p0 \
  --harmonized-root "$harmonized_root" \
  --features-root "$features_root" \
  --manifest "$manifest_path" \
  --condition-feature-top-n "$CONDITION_FEATURE_TOP_N" \
  --trend-min-events "$TREND_MIN_EVENTS" \
  --stay-feature-batches "$STAY_FEATURE_BATCHES" \
  --event-sequence-batches "$EVENT_SEQUENCE_BATCHES"; then
  features_rc=0
else
  features_rc=$?
fi
echo "=== phase8_p0_features done exit=$features_rc ==="

if (( features_rc == 0 )); then
  echo "Review aggregate report:"
  echo "  $manifest_path"
  echo "Local patient-level artifacts stay ignored under $features_root/"
fi

exit "$features_rc"
