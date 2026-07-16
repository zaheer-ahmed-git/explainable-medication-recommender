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
# Runs the full Milestone 6 preprocessing materialization in one job:
# pipeline.features, pipeline.build_training_table, then pipeline.preprocessing.
# Later steps depend on earlier artifacts, so this wrapper stops on the first
# failure. Run only after Milestone 5 harmonization and its aggregate coverage
# reports are reviewed.

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
    (( DUCKDB_THREADS > 4 )) && DUCKDB_THREADS=4
  fi
  : "${DUCKDB_THREADS:=4}"
  export DUCKDB_THREADS
fi
if [[ -z "${DUCKDB_MEMORY_LIMIT:-}" ]]; then
  export DUCKDB_MEMORY_LIMIT="10GB"
fi
: "${STAY_FEATURE_BATCHES:=8}"
export STAY_FEATURE_BATCHES
: "${EVENT_SEQUENCE_BATCHES:=8}"
export EVENT_SEQUENCE_BATCHES
: "${CANDIDATE_TOKEN_STRATEGY:=rxnorm_or_atc}"
export CANDIDATE_TOKEN_STRATEGY
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"
echo "STAY_FEATURE_BATCHES=${STAY_FEATURE_BATCHES:-}"
echo "EVENT_SEQUENCE_BATCHES=${EVENT_SEQUENCE_BATCHES:-}"
echo "CANDIDATE_TOKEN_STRATEGY=${CANDIDATE_TOKEN_STRATEGY:-}"

echo "=== features start ==="
if uv run python -m pipeline.features \
  --stay-feature-batches "$STAY_FEATURE_BATCHES" \
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
if uv run python -m pipeline.build_training_table \
  --candidate-token-strategy "$CANDIDATE_TOKEN_STRATEGY"; then
  training_rc=0
else
  training_rc=$?
fi
echo "=== build_training_table done exit=$training_rc ==="
if (( training_rc != 0 )); then
  echo "Training table build failed (exit=$training_rc); skipping preprocessing." >&2
  exit "$training_rc"
fi

echo "=== preprocessing start ==="
if uv run python -m pipeline.preprocessing; then
  preprocessing_rc=0
else
  preprocessing_rc=$?
fi
echo "=== preprocessing done exit=$preprocessing_rc ==="

if (( preprocessing_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  milestone6_feature_manifest.json"
  echo "  training_table_manifest.json"
  echo "  preprocessing_manifest.json"
fi

exit "$preprocessing_rc"
