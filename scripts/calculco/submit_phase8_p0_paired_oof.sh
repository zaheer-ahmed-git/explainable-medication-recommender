#!/usr/bin/env bash
# Submit the two independent paired-OOF arrays or a later serial boundary.
#
# Usage:
#   scripts/calculco/submit_phase8_p0_paired_oof.sh all-oof
#   scripts/calculco/submit_phase8_p0_paired_oof.sh transformer-fold FOLD
#   scripts/calculco/submit_phase8_p0_paired_oof.sh gnn-variant VARIANT
#   scripts/calculco/submit_phase8_p0_paired_oof.sh gnn-chain-remaining
#   scripts/calculco/submit_phase8_p0_paired_oof.sh {transformer|gnn|select|refit|score}

set -euo pipefail
umask 077

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_home="${PROJECT_HOME:-$(cd "$script_dir/../.." && pwd)}"
action="${1:-all-oof}"
item="${2:-}"
: "${WORK_SCRATCH:?Export WORK_SCRATCH before submitting paired OOF jobs}"

case "$WORK_SCRATCH" in
  /path|/path/*)
    echo "WORK_SCRATCH looks like a documentation placeholder." >&2
    exit 2
    ;;
esac

job_env="$script_dir/phase8_p0_paired_oof_job.env"
{
  printf 'WORK_SCRATCH=%q\n' "$WORK_SCRATCH"
  printf 'PAIRED_SEED=%q\n' "${PAIRED_SEED:-20260617}"
  printf 'PAIRED_FOLD_COUNT=5\n'
  printf 'PAIRED_NEURAL_SHARD_COUNT=%q\n' "${PAIRED_NEURAL_SHARD_COUNT:-8}"
  printf 'PAIRED_GNN_SHARD_COUNT=%q\n' "${PAIRED_GNN_SHARD_COUNT:-256}"
  printf 'PAIRED_DEVICE=%q\n' "${PAIRED_DEVICE:-cuda}"
  printf 'PAIRED_GNN_PRECISION=%q\n' "${PAIRED_GNN_PRECISION:-bf16}"
  printf 'PAIRED_TRANSFORMER_FIXED_EPOCHS=%q\n' "${PAIRED_TRANSFORMER_FIXED_EPOCHS:-}"
  printf 'PAIRED_GATE_SCORE_CONFIRM=%q\n' "${PAIRED_GATE_SCORE_CONFIRM:-}"
  printf 'DUCKDB_THREADS=%q\n' "${DUCKDB_THREADS:-8}"
  printf 'DUCKDB_MEMORY_LIMIT=%q\n' "${DUCKDB_MEMORY_LIMIT:-24GB}"
} >"$job_env"
chmod 600 "$job_env"

mkdir -p "$project_home/scripts/calculco/logs"

dependency_args=()
if [[ -n "${PAIRED_AFTER_JOB_ID:-}" ]]; then
  if [[ ! "$PAIRED_AFTER_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "PAIRED_AFTER_JOB_ID must be a numeric OAR job identifier." >&2
    exit 2
  fi
  dependency_args=(-a "$PAIRED_AFTER_JOB_ID")
fi

notify_args=()
if [[ -n "${PAIRED_OAR_NOTIFY:-}" ]]; then
  notify_args=(--notify "$PAIRED_OAR_NOTIFY")
  echo "OAR notify: $PAIRED_OAR_NOTIFY"
fi

submit_array() {
  local name="$1"
  local parameter_file="$2"
  local output=""
  if ! output="$(
    oarsub \
      "${dependency_args[@]}" \
      "${notify_args[@]}" \
      --array-param-file "$parameter_file" \
      -O "$project_home/scripts/calculco/logs/rm_phase8_p0_paired_${name}_%jobid%.out" \
      -E "$project_home/scripts/calculco/logs/rm_phase8_p0_paired_${name}_%jobid%.err" \
      -S "$script_dir/phase8_p0_paired_oof.sh" 2>&1
  )"; then
    echo "$output" >&2
    return 1
  fi
  printf '%s\n' "$output"
  local job_id=""
  if job_id="$("$script_dir/oar_job_id_from_output.sh" <<<"$output")"; then
    echo "Submitted paired OOF ${name} as OAR job ${job_id}"
  fi
}

submit_transformer() {
  local parameter_file="$WORK_SCRATCH/phase8_p0_paired_transformer.params"
  : >"$parameter_file"
  for fold in 0 1 2 3 4; do
    printf 'transformer-fold %s\n' "$fold" >>"$parameter_file"
  done
  submit_array transformer "$parameter_file"
}

submit_gnn() {
  local parameter_file="$WORK_SCRATCH/phase8_p0_paired_gnn.params"
  : >"$parameter_file"
  for variant in full rank_only no_message_passing no_condition_medication no_dense_lab_vital no_lab_vital_intervention; do
    printf 'gnn-variant %s\n' "$variant" >>"$parameter_file"
  done
  submit_array gnn "$parameter_file"
}

submit_single() {
  local task="$1"
  local parameter_file="$WORK_SCRATCH/phase8_p0_paired_${task}.params"
  printf '%s\n' "$task" >"$parameter_file"
  submit_array "$task" "$parameter_file"
}

submit_targeted() {
  local name="$1"
  local task="$2"
  local task_item="$3"
  local parameter_file="$WORK_SCRATCH/phase8_p0_paired_${name}.params"
  printf '%s %s\n' "$task" "$task_item" >"$parameter_file"
  submit_array "$name" "$parameter_file"
}

submit_select() {
  oarsub \
    -O "$project_home/scripts/calculco/logs/rm_phase8_p0_paired_select_%jobid%.out" \
    -E "$project_home/scripts/calculco/logs/rm_phase8_p0_paired_select_%jobid%.err" \
    -S "$script_dir/phase8_p0_paired_oof_select.sh"
}

case "$action" in
  all-oof)
    # Independent submissions allow the five Transformer folds and six GNN
    # materializations to occupy all scheduler-available GPUs concurrently.
    submit_transformer
    submit_gnn
    ;;
  transformer) submit_transformer ;;
  gnn) submit_gnn ;;
  transformer-fold)
    if [[ ! "$item" =~ ^[0-4]$ ]]; then
      echo "transformer-fold requires a fold in 0..4." >&2
      exit 2
    fi
    submit_targeted "transformer_retry_fold_${item}" transformer-fold "$item"
    ;;
  gnn-variant)
    case "$item" in
      full|rank_only|no_message_passing|no_condition_medication|no_dense_lab_vital|no_lab_vital_intervention) ;;
      *)
        echo "gnn-variant requires a registered variant name." >&2
        exit 2
        ;;
    esac
    submit_targeted "gnn_retry_${item}" gnn-variant "$item"
    ;;
  gnn-chain-remaining)
    chain_args=()
    if [[ -n "${PAIRED_GNN_CHAIN_MAIL:-${OAR_WATCH_MAIL:-}}" ]]; then
      chain_args+=(--mail "${PAIRED_GNN_CHAIN_MAIL:-${OAR_WATCH_MAIL:-}}")
    fi
    if [[ -n "${PAIRED_GNN_CHAIN_ON_COMPLETE:-}" ]]; then
      chain_args+=(--on-chain-complete "$PAIRED_GNN_CHAIN_ON_COMPLETE")
    fi
    exec "$script_dir/watch_paired_gnn_oof_chain.sh" "${chain_args[@]}"
    ;;
  select) submit_select ;;
  refit) submit_single refit ;;
  score)
    if [[ "${PAIRED_GATE_SCORE_CONFIRM:-}" != "I_UNDERSTAND_ONE_SHOT" ]]; then
      echo "Refusing one-shot gate submission without explicit confirmation." >&2
      exit 2
    fi
    submit_single score
    ;;
  *)
    echo "Usage: $0 {all-oof|transformer|gnn|transformer-fold FOLD|gnn-variant VARIANT|gnn-chain-remaining|select|refit|score}" >&2
    exit 2
    ;;
esac
