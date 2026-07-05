#!/bin/bash
#OAR -n rm_milestone6
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# During the 2026 Calculco migration, legacy CPU nodes on the calculco front-end
# may be offline while GPU nodes remain Alive. If submission fails with "not
# enough resources", SSH to ritchie.univ-littoral.fr (chimay CPU nodes) and
# uncomment the gpudevice line above. Do not prefix disabled directives with #OAR;
# oarsub parses every #OAR line in this file.
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/milestone6.sh"
#
# Runs the full Milestone 6 materialization in one job: pipeline.features then
# pipeline.build_training_table. build_training_table depends on the feature
# artifacts, so this wrapper stops if the feature step fails. Run only after
# Milestone 5 harmonization and its aggregate coverage reports are reviewed.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

harmonized_root="$DATASET_ROOT/processed/harmonized"

echo "=== milestone6 preflight ==="
preflight_fail=0
for tbl in cohort_stays demographics conditions medications labs vitals \
  allergies interventions temporal_events; do
  if [[ -f "$harmonized_root/$tbl.parquet" ]]; then
    echo "preflight ok: $tbl"
  else
    echo "preflight MISSING: $tbl ($harmonized_root/$tbl.parquet)" >&2
    preflight_fail=1
  fi
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; run harmonization before Milestone 6." >&2
  exit 1
fi

# Bound DuckDB threads/memory to the OAR allocation (large temporal_events,
# vitals, and medications scans). Override by exporting before submit.
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
: "${EVENT_SEQUENCE_BATCHES:=8}"
export EVENT_SEQUENCE_BATCHES
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"
echo "EVENT_SEQUENCE_BATCHES=${EVENT_SEQUENCE_BATCHES:-}"

echo "=== features start ==="
if uv run python -m pipeline.features \
  --event-sequence-batches "$EVENT_SEQUENCE_BATCHES"; then
  features_rc=0
else
  features_rc=$?
fi
echo "=== features done exit=$features_rc ==="
if (( features_rc != 0 )); then
  echo "Feature build failed (exit=$features_rc); skipping training table." >&2
  exit "$features_rc"
fi

echo "=== build_training_table start ==="
if uv run python -m pipeline.build_training_table; then
  training_rc=0
else
  training_rc=$?
fi
echo "=== build_training_table done exit=$training_rc ==="

if (( training_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  milestone6_feature_manifest.json"
  echo "  training_table_manifest.json"
fi

exit "$training_rc"
