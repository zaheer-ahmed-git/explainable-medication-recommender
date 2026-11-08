#!/bin/bash
#OAR -n rm_phase8_p0_model_ready
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_model_ready_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_model_ready_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_model_ready.sh"
#
# Chains the complete Phase 8 P0 model-ready package under
# $DATASET_ROOT/processed/phase8_p0. Local artifacts remain ignored; reports
# contain aggregate metadata or schemas only.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

# OAR jobs run in a clean environment and do not inherit login-shell exports.
# Source a gitignored job env file (written by submit_phase8_p0_model_ready.sh)
# so resume controls like PHASE8_P0_START_AT and the subgraph knobs reach the
# worker instead of silently falling back to script defaults.
phase8_job_env="${PHASE8_P0_MODEL_READY_JOB_ENV:-$script_dir/phase8_p0_model_ready_job.env}"
if [[ -f "$phase8_job_env" ]]; then
  echo "=== phase8_p0_model_ready_job.env ==="
  set -a
  # shellcheck disable=SC1090
  source "$phase8_job_env"
  set +a
  cat "$phase8_job_env"
fi

mkdir -p "$script_dir/logs"

harmonized_root="${HARMONIZED_ROOT_OVERRIDE:-$DATASET_ROOT/processed/harmonized}"
phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
preprocessing_root="${PHASE8_P0_PREPROCESSING_ROOT:-$training_root/preprocessing}"
sensitivity_training_root="${PHASE8_P0_ATC3_TRAINING_ROOT:-$phase8_root/sensitivity/atc3_or_rxnorm/training}"
graph_root="${PHASE8_P0_GRAPH_ROOT:-$phase8_root/graph/milestone8}"
subgraphs_root="${PHASE8_P0_SUBGRAPHS_ROOT:-$graph_root/patient_subgraphs}"
package_root="${PHASE8_P0_PACKAGE_ROOT:-$phase8_root/model_ready}"
feature_manifest="${PHASE8_P0_FEATURE_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_milestone6_feature_manifest.json}"
training_manifest="${PHASE8_P0_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_training_table_manifest.json}"
atc3_training_manifest="${PHASE8_P0_ATC3_TRAINING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_atc3_training_table_manifest.json}"
preprocessing_manifest="${PHASE8_P0_PREPROCESSING_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_preprocessing_manifest.json}"
graph_schema_report="${PHASE8_P0_GRAPH_SCHEMA_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8_graph_schema.json}"
graph_suitability_report="${PHASE8_P0_GRAPH_SUITABILITY_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8_graph_suitability.json}"
graph_ablation_plan="${PHASE8_P0_GRAPH_ABLATION_PLAN:-$PROJECT_HOME/reports/phase8_p0_milestone8_ablation_plan.json}"
subgraphs_manifest="${PHASE8_P0_SUBGRAPHS_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_patient_subgraphs_manifest.json}"
data_dictionary="${PHASE8_P0_DATA_DICTIONARY:-$PROJECT_HOME/reports/phase8_p0_model_ready_data_dictionary.json}"
model_ready_manifest="${PHASE8_P0_MODEL_READY_MANIFEST:-$PROJECT_HOME/reports/phase8_p0_model_ready_manifest.json}"
: "${PHASE8_P0_START_AT:=training}"

case "$PHASE8_P0_START_AT" in
  training|subgraphs) ;;
  *)
    echo "PHASE8_P0_START_AT must be 'training' or 'subgraphs'." >&2
    exit 2
    ;;
esac

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
if [[ "$PHASE8_P0_START_AT" == "training" ]]; then
  for tbl in cohort_stays demographics conditions medications labs vitals allergies interventions temporal_events; do
    check_file "$tbl" "$harmonized_root/$tbl.parquet"
  done
else
  check_file "cohort_decision_times" "$features_root/cohort_decision_times.parquet"
  check_file "patient_stay_features" "$features_root/patient_stay_features.parquet"
  check_file "event_sequences" "$features_root/event_sequences.parquet"
  check_file "model-ready cohort_stays" "$training_root/cohort_stays.parquet"
  check_file "patient_condition_medication" "$training_root/patient_condition_medication.parquet"
  check_file "split_manifest" "$training_root/split_manifest.parquet"
  check_file "candidate_catalog" "$training_root/candidate_catalog.parquet"
  check_file "graph_edges" "$graph_root/graph_edges.parquet"
  check_file "train-fitted preprocessor" "$preprocessing_root/train_fitted_preprocessor.joblib"
  check_file "primary training manifest" "$training_manifest"
  check_file "ATC3 sensitivity manifest" "$atc3_training_manifest"
  check_file "preprocessing manifest" "$preprocessing_manifest"
fi
if (( preflight_fail != 0 )); then
  echo "Preflight failed for PHASE8_P0_START_AT=$PHASE8_P0_START_AT." >&2
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
# Optional cap on DuckDB spill size. DuckDB otherwise uses ~90% of free space on
# the temp drive, so a small node-local /tmp silently limits spilling and raises
# "failed to offload data block". Export DUCKDB_MAX_TEMP_DIR_SIZE (e.g. 150GB)
# when DUCKDB_TEMP_DIR points at a larger volume. pipeline.config reads it.
if [[ -n "${DUCKDB_MAX_TEMP_DIR_SIZE:-}" ]]; then
  export DUCKDB_MAX_TEMP_DIR_SIZE
fi
: "${STAY_FEATURE_BATCHES:=8}"
: "${EVENT_SEQUENCE_BATCHES:=8}"
: "${SUBGRAPH_BATCHES:=8}"
: "${SUBGRAPH_JOIN_SHARDS:=8}"
: "${SUBGRAPH_EDGE_THREADS:=1}"
: "${CONDITION_FEATURE_TOP_N:=40}"
: "${TREND_MIN_EVENTS:=2}"
: "${CANDIDATE_TOKEN_STRATEGY:=rxnorm_or_atc}"
: "${PHASE8_P0_FEATURE_VERSION:=temporal-features-v2}"
: "${PHASE8_P0_RUN_FEATURES:=auto}"
export STAY_FEATURE_BATCHES EVENT_SEQUENCE_BATCHES SUBGRAPH_BATCHES
export SUBGRAPH_JOIN_SHARDS SUBGRAPH_EDGE_THREADS
export CONDITION_FEATURE_TOP_N
export TREND_MIN_EVENTS CANDIDATE_TOKEN_STRATEGY

echo "harmonized_root=$harmonized_root"
echo "features_root=$features_root"
echo "training_root=$training_root"
echo "preprocessing_root=$preprocessing_root"
echo "sensitivity_training_root=$sensitivity_training_root"
echo "graph_root=$graph_root"
echo "subgraphs_root=$subgraphs_root"
echo "package_root=$package_root"
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS:-}"
echo "DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-}"
echo "DUCKDB_MAX_TEMP_DIR_SIZE=${DUCKDB_MAX_TEMP_DIR_SIZE:-}"
echo "SUBGRAPH_BATCHES=$SUBGRAPH_BATCHES"
echo "SUBGRAPH_JOIN_SHARDS=$SUBGRAPH_JOIN_SHARDS"
echo "SUBGRAPH_EDGE_THREADS=$SUBGRAPH_EDGE_THREADS"
echo "PHASE8_P0_START_AT=$PHASE8_P0_START_AT"
echo "PHASE8_P0_RUN_FEATURES=$PHASE8_P0_RUN_FEATURES"
echo "CANDIDATE_TOKEN_STRATEGY=$CANDIDATE_TOKEN_STRATEGY"
echo "PHASE8_P0_FEATURE_VERSION=$PHASE8_P0_FEATURE_VERSION"

if [[ "$CANDIDATE_TOKEN_STRATEGY" != "rxnorm_or_atc" ]]; then
  echo "Phase 8 P0 completion requires primary CANDIDATE_TOKEN_STRATEGY=rxnorm_or_atc." >&2
  exit 1
fi

if [[ "$PHASE8_P0_START_AT" == "training" ]]; then
run_features=0
if [[ "$PHASE8_P0_RUN_FEATURES" == "1" ]]; then
  run_features=1
elif [[ "$PHASE8_P0_RUN_FEATURES" == "auto" ]]; then
  for feature_table in cohort_decision_times patient_stay_features event_sequences; do
    if [[ ! -f "$features_root/$feature_table.parquet" ]]; then
      run_features=1
    fi
  done
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
  --feature-version "$PHASE8_P0_FEATURE_VERSION" \
  --candidate-token-strategy "$CANDIDATE_TOKEN_STRATEGY"; then
  training_rc=0
else
  training_rc=$?
fi
echo "=== phase8_p0_training_table done exit=$training_rc ==="
if (( training_rc != 0 )); then
  exit "$training_rc"
fi

echo "=== phase8_p0_atc3_sensitivity start ==="
if uv run python -m pipeline.build_training_table \
  --harmonized-root "$harmonized_root" \
  --features-root "$features_root" \
  --training-root "$sensitivity_training_root" \
  --manifest "$atc3_training_manifest" \
  --feature-version "$PHASE8_P0_FEATURE_VERSION" \
  --candidate-token-strategy atc3_or_rxnorm; then
  sensitivity_rc=0
else
  sensitivity_rc=$?
fi
echo "=== phase8_p0_atc3_sensitivity done exit=$sensitivity_rc ==="
if (( sensitivity_rc != 0 )); then
  exit "$sensitivity_rc"
fi

echo "=== phase8_p0_preprocessing start ==="
if uv run python -m pipeline.preprocessing \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --preprocessing-root "$preprocessing_root" \
  --manifest "$preprocessing_manifest" \
  --feature-version "$PHASE8_P0_FEATURE_VERSION"; then
  preprocessing_rc=0
else
  preprocessing_rc=$?
fi
echo "=== phase8_p0_preprocessing done exit=$preprocessing_rc ==="
if (( preprocessing_rc != 0 )); then
  exit "$preprocessing_rc"
fi

echo "=== phase8_p0_graph_suitability start ==="
if uv run python -m pipeline.graph_suitability \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-root "$graph_root" \
  --graph-schema-report "$graph_schema_report" \
  --suitability-report "$graph_suitability_report" \
  --ablation-plan "$graph_ablation_plan" \
  --feature-version "$PHASE8_P0_FEATURE_VERSION"; then
  graph_rc=0
else
  graph_rc=$?
fi
echo "=== phase8_p0_graph_suitability done exit=$graph_rc ==="
if (( graph_rc != 0 )); then
  exit "$graph_rc"
fi
else
  echo "Skipping completed training, ATC3, preprocessing, and graph stages."
fi

echo "=== phase8_p0_patient_subgraphs start ==="
if uv run python -m pipeline.patient_subgraphs \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-root "$graph_root" \
  --subgraphs-root "$subgraphs_root" \
  --manifest "$subgraphs_manifest" \
  --feature-version "$PHASE8_P0_FEATURE_VERSION" \
  --subgraph-batches "$SUBGRAPH_BATCHES" \
  --subgraph-join-shards "$SUBGRAPH_JOIN_SHARDS" \
  --edge-duckdb-threads "$SUBGRAPH_EDGE_THREADS"; then
  subgraphs_rc=0
else
  subgraphs_rc=$?
fi
echo "=== phase8_p0_patient_subgraphs done exit=$subgraphs_rc ==="
if (( subgraphs_rc != 0 )); then
  exit "$subgraphs_rc"
fi

echo "=== phase8_p0_model_ready_package start ==="
if uv run python -m pipeline.model_ready_package \
  --features-root "$features_root" \
  --training-root "$training_root" \
  --graph-root "$graph_root" \
  --subgraphs-root "$subgraphs_root" \
  --preprocessing-root "$preprocessing_root" \
  --package-root "$package_root" \
  --data-dictionary "$data_dictionary" \
  --manifest "$model_ready_manifest" \
  --primary-training-manifest "$training_manifest" \
  --sensitivity-training-manifest "$atc3_training_manifest" \
  --preprocessing-manifest "$preprocessing_manifest" \
  --subgraphs-manifest "$subgraphs_manifest" \
  --feature-version "$PHASE8_P0_FEATURE_VERSION"; then
  package_rc=0
else
  package_rc=$?
fi
echo "=== phase8_p0_model_ready_package done exit=$package_rc ==="

if (( package_rc == 0 )); then
  echo "Review aggregate reports:"
  echo "  $feature_manifest"
  echo "  $training_manifest"
  echo "  $atc3_training_manifest"
  echo "  $preprocessing_manifest"
  echo "  $graph_suitability_report"
  echo "  $subgraphs_manifest"
  echo "  $data_dictionary"
  echo "  $model_ready_manifest"
  echo "Local patient-level artifacts stay ignored under $phase8_root/"
fi

exit "$package_rc"
