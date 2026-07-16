#!/bin/bash
#OAR -n rm_training_table
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# During the 2026 Calculco migration, legacy CPU nodes on the calculco front-end
# may be offline while GPU nodes remain Alive. If submission fails with "not
# enough resources", SSH to ritchie.univ-littoral.fr (chimay CPU nodes) and
# uncomment the gpudevice line above. Do not prefix disabled directives with #OAR;
# oarsub parses every #OAR line in this file.
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/build_training_table.sh"
#
# Builds Milestone 6 training artifacts (split_manifest, candidate_catalog,
# patient_condition_medication) from harmonized tables and the feature
# artifacts produced by features.sh. Candidate catalogs are learned from MIMIC
# train positives only. Writes ignored artifacts under
# $DATASET_ROOT/processed/training/ and the aggregate-only manifest
# reports/training_table_manifest.json.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

harmonized_root="$DATASET_ROOT/processed/harmonized"
features_root="$DATASET_ROOT/processed/features"

echo "=== build_training_table preflight ==="
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
# Required harmonized inputs (REQUIRED_HARMONIZED_TABLES).
for tbl in conditions medications; do
  check_file "harmonized $tbl" "$harmonized_root/$tbl.parquet"
done
# Required feature inputs (REQUIRED_FEATURE_TABLES) from features.sh.
for tbl in cohort_decision_times patient_stay_features; do
  check_file "feature $tbl" "$features_root/$tbl.parquet"
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; run harmonization and features.sh first." >&2
  exit 1
fi

echo "harmonized_root=$harmonized_root"
echo "features_root=$features_root"

# Bound DuckDB threads/memory to the OAR allocation. The label-window medication
# join scans the multi-GB harmonized medications table and the stay x condition
# x candidate cross-product can be large, so spill to DUCKDB_TEMP_DIR instead of
# being SIGKILLed by the cgroup. Override by exporting before submit.
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
: "${CANDIDATE_TOKEN_STRATEGY:=rxnorm_or_atc}"
export CANDIDATE_TOKEN_STRATEGY
echo "CANDIDATE_TOKEN_STRATEGY=${CANDIDATE_TOKEN_STRATEGY:-}"

echo "=== build_training_table start ==="
if uv run python -m pipeline.build_training_table \
  --candidate-token-strategy "$CANDIDATE_TOKEN_STRATEGY"; then
  training_rc=0
else
  training_rc=$?
fi
echo "=== build_training_table done exit=$training_rc ==="

if (( training_rc == 0 )); then
  echo "Review aggregate report under $PROJECT_HOME/reports/:"
  echo "  training_table_manifest.json (split_integrity, candidate_catalog_counts,"
  echo "  training_rows_by_source_split, out_of_catalog_positives, coverage losses)"
  echo "Run uv run python -m pipeline.preprocessing to fit train-only preprocessing artifacts."
fi

exit "$training_rc"
