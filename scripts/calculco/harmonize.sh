#!/bin/bash
#OAR -n rm_harmonize
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# During the 2026 Calculco migration, legacy CPU nodes on the calculco front-end
# may be offline while GPU nodes remain Alive. If submission fails with "not
# enough resources", SSH to ritchie.univ-littoral.fr (chimay CPU nodes) and
# uncomment the gpudevice line above. Do not prefix disabled directives with #OAR;
# oarsub parses every #OAR line in this file.
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/harmonize.sh"

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

extracts_root="$DATASET_ROOT/processed/extracts"
mapping_root="$DATASET_ROOT/mappings"
harmonized_root="$DATASET_ROOT/processed/harmonized"

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

echo "=== harmonize preflight ==="
check_file "cohort_stays" "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
check_file "mimic diagnoses_icd" "$extracts_root/mimiciv/diagnoses_icd.parquet"
check_file "eicu diagnosis" "$extracts_root/eicu_crd/diagnosis.parquet"
check_file "mimic_ndc_rxnorm_atc" "$mapping_root/medications/mimic_ndc_rxnorm_atc.csv"
check_file "eicu_drug_rxnorm_atc" "$mapping_root/medications/eicu_drug_rxnorm_atc.csv"

if [[ -f "$mapping_root/conditions/icd10_ccsr.csv" ]]; then
  echo "preflight ok: condition icd10_ccsr (optional roll-up)"
else
  echo "preflight note: condition icd10_ccsr.csv missing; coded rows degrade to category fallback"
fi
if [[ -f "$mapping_root/conditions/icd9_ccs.csv" ]]; then
  echo "preflight ok: condition icd9_ccs (optional roll-up)"
else
  echo "preflight note: condition icd9_ccs.csv missing; coded rows degrade to category fallback"
fi

if (( preflight_fail != 0 )); then
  echo "Preflight failed; fix missing inputs before harmonization." >&2
  exit 1
fi

echo "extracts_root=$extracts_root"
echo "harmonized_root=$harmonized_root"
echo "mapping_root=$mapping_root"

# Bound DuckDB threads/memory to the OAR allocation so the large eICU vital
# fan-out (COPY over a 7-column UNION of vital_periodic) spills to DUCKDB_TEMP_DIR
# instead of being SIGKILLed by the cgroup. Override by exporting before submit.
if [[ -z "${DUCKDB_THREADS:-}" ]]; then
  if [[ -n "${OAR_NODE_FILE:-}" && -f "${OAR_NODE_FILE}" ]]; then
    DUCKDB_THREADS="$(wc -l < "$OAR_NODE_FILE" | tr -d '[:space:]')"
  fi
  : "${DUCKDB_THREADS:=4}"
  export DUCKDB_THREADS
fi
if [[ -z "${DUCKDB_MEMORY_LIMIT:-}" ]]; then
  # ~3 GB per allocated core, leaving headroom for Python/OS inside the cgroup.
  mem_gb=$(( DUCKDB_THREADS * 3 ))
  (( mem_gb < 6 )) && mem_gb=6
  export DUCKDB_MEMORY_LIMIT="${mem_gb}GB"
fi
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"

echo "=== harmonize start ==="
if uv run python -m pipeline.harmonize; then
  harmonize_rc=0
else
  harmonize_rc=$?
fi
echo "=== harmonize done exit=$harmonize_rc ==="

if (( harmonize_rc == 0 )); then
  echo "Review aggregate reports under $PROJECT_HOME/reports/:"
  echo "  harmonization_manifest.json"
  echo "  harmonization_coverage.json"
  echo "  condition_normalization_coverage.json"
  echo "  unmapped_concepts.json"
fi

exit "$harmonize_rc"
