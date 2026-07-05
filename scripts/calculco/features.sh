#!/bin/bash
#OAR -n rm_features
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# During the 2026 Calculco migration, legacy CPU nodes on the calculco front-end
# may be offline while GPU nodes remain Alive. If submission fails with "not
# enough resources", SSH to ritchie.univ-littoral.fr (chimay CPU nodes) and
# uncomment the gpudevice line above. Do not prefix disabled directives with #OAR;
# oarsub parses every #OAR line in this file.
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/features.sh"
#
# Builds Milestone 6 temporal feature artifacts (cohort_decision_times,
# patient_stay_features, event_sequences) from harmonized tables. Run only
# after Milestone 5 harmonization and its aggregate coverage reports have been
# reviewed. Writes ignored artifacts under
# $DATASET_ROOT/processed/features/ and the aggregate-only manifest
# reports/milestone6_feature_manifest.json.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

harmonized_root="$DATASET_ROOT/processed/harmonized"
features_root="$DATASET_ROOT/processed/features"

echo "=== features preflight ==="
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
# Required harmonized inputs for pipeline.features (see REQUIRED_HARMONIZED_TABLES).
for tbl in cohort_stays demographics labs vitals allergies interventions temporal_events; do
  check_file "$tbl" "$harmonized_root/$tbl.parquet"
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; run harmonization before feature construction." >&2
  exit 1
fi

echo "harmonized_root=$harmonized_root"
echo "features_root=$features_root"

# Bound DuckDB threads/memory to the OAR allocation. event_sequences scans the
# multi-GB harmonized temporal_events table, and the lab/vital aggregates fan
# out over large event tables, so spill to DUCKDB_TEMP_DIR instead of being
# SIGKILLed by the cgroup. Override by exporting before submit.
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

echo "=== features start ==="
if uv run python -m pipeline.features; then
  features_rc=0
else
  features_rc=$?
fi
echo "=== features done exit=$features_rc ==="

if (( features_rc == 0 )); then
  echo "Review aggregate report under $PROJECT_HOME/reports/:"
  echo "  milestone6_feature_manifest.json (eligibility_counts, split_counts, temporal_event_exclusions)"
fi

exit "$features_rc"
