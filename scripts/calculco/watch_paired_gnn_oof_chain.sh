#!/usr/bin/env bash
# Serial paired-OOF GNN variant chain: submit, watch, verify report, repeat.
#
# Skips variants whose aggregate report already says status=completed.
# Sends mail on every job end and again on chain failure. Does not read patient rows.
#
# Usage (run inside tmux/screen on the login node):
#   export PROJECT_HOME DATASET_ROOT WORK_SCRATCH
#   scripts/calculco/watch_paired_gnn_oof_chain.sh --mail you@example.com
#
# Attach to an already submitted variant job, then continue the queue:
#   scripts/calculco/watch_paired_gnn_oof_chain.sh --from-job 22500 --variant rank_only \
#     --mail you@example.com
#
# After the last variant completes, optionally submit CPU selection:
#   scripts/calculco/watch_paired_gnn_oof_chain.sh --mail you@example.com \
#     --on-chain-complete 'scripts/calculco/submit_phase8_p0_paired_oof.sh select'

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_home="${PROJECT_HOME:-$(cd "$script_dir/../.." && pwd)}"
: "${WORK_SCRATCH:?Export WORK_SCRATCH before starting the paired GNN chain}"

mail_to="${PAIRED_GNN_CHAIN_MAIL:-${OAR_WATCH_MAIL:-${GNN_WATCH_MAIL:-}}}"
interval_sec="${PAIRED_GNN_CHAIN_INTERVAL_SEC:-${OAR_WATCH_INTERVAL_SEC:-120}}"
from_job=""
from_variant=""
on_chain_complete=""

ALL_VARIANTS=(
  full
  rank_only
  no_message_passing
  no_condition_medication
  no_dense_lab_vital
  no_lab_vital_intervention
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail)
      mail_to="${2:?--mail requires an address}"
      shift 2
      ;;
    --from-job)
      from_job="${2:?--from-job requires a numeric OAR job id}"
      shift 2
      ;;
    --variant)
      from_variant="${2:?--variant requires a registered GNN ablation name}"
      shift 2
      ;;
    --on-chain-complete)
      on_chain_complete="${2:?--on-chain-complete requires a command}"
      shift 2
      ;;
    --interval)
      interval_sec="${2:?--interval requires seconds}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$from_job" && ! "$from_job" =~ ^[0-9]+$ ]]; then
  echo "--from-job must be numeric." >&2
  exit 2
fi

report_path_for_variant() {
  printf '%s/reports/phase8_p0_gnn_paired_oof_%s.json' "$project_home" "$1"
}

variant_report_status() {
  local variant="$1"
  local report_path
  report_path="$(report_path_for_variant "$variant")"
  if [[ ! -f "$report_path" ]]; then
    echo "missing"
    return 0
  fi
  python3 - "$report_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("unreadable")
    raise SystemExit(0)
print(payload.get("status") or "unknown")
PY
}

variant_report_reason() {
  local variant="$1"
  local report_path
  report_path="$(report_path_for_variant "$variant")"
  [[ -f "$report_path" ]] || return 0
  python3 - "$report_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
reason = payload.get("reason")
if isinstance(reason, str) and reason.strip():
    print(reason.strip())
PY
}

pending_variants() {
  local variant
  for variant in "${ALL_VARIANTS[@]}"; do
    if [[ "$(variant_report_status "$variant")" == "completed" ]]; then
      continue
    fi
    printf '%s\n' "$variant"
  done
}

submit_variant() {
  local variant="$1"
  local output=""
  local job_id=""
  if ! output="$(
    PROJECT_HOME="$project_home" WORK_SCRATCH="$WORK_SCRATCH" \
      "$script_dir/submit_phase8_p0_paired_oof.sh" gnn-variant "$variant" 2>&1
  )"; then
    printf '%s\n' "$output" >&2
    return 1
  fi
  printf '%s\n' "$output" >&2
  if ! job_id="$("$script_dir/oar_job_id_from_output.sh" <<<"$output")"; then
    return 1
  fi
  printf '%s\n' "$job_id"
}

watch_variant_job() {
  local variant="$1"
  local job_id="$2"
  local watch_output=""
  local watch_args=(--interval "$interval_sec")
  if [[ -n "$mail_to" ]]; then
    watch_args+=(--mail "$mail_to")
  fi
  if ! watch_output="$(
    OAR_WATCH_INTERVAL_SEC="$interval_sec" \
      "$script_dir/watch_oar_job.sh" "$job_id" "${watch_args[@]}" 2>&1
  )"; then
    printf '%s\n' "$watch_output"
    return 1
  fi
  printf '%s\n' "$watch_output"

  local report_status
  report_status="$(variant_report_status "$variant")"
  if [[ "$report_status" != "completed" ]]; then
    local reason
    reason="$(variant_report_reason "$variant")"
    echo "Aggregate report for ${variant} is not completed (status=${report_status})." >&2
    if [[ -n "$reason" ]]; then
      echo "Report reason: $reason" >&2
    fi
    if [[ -n "$mail_to" ]]; then
      "$script_dir/send_oar_mail.sh" "$mail_to" \
        "Paired GNN chain stopped: ${variant} report=${report_status}" \
        "OAR job ${job_id} ended but reports/phase8_p0_gnn_paired_oof_${variant}.json has status=${report_status}.
${reason:+$reason

}Fix the failure, then restart with:
  scripts/calculco/watch_paired_gnn_oof_chain.sh --mail ${mail_to}"
    fi
    return 1
  fi
  return 0
}

mapfile -t queue < <(pending_variants)
if [[ -n "$from_job" ]]; then
  if [[ -z "$from_variant" ]]; then
    echo "--from-job requires --variant so the chain can verify the aggregate report." >&2
    exit 2
  fi
  filtered=()
  skip=1
  for variant in "${queue[@]}"; do
    if [[ "$skip" -eq 1 && "$variant" != "$from_variant" ]]; then
      continue
    fi
    skip=0
    filtered+=("$variant")
  done
  queue=("${filtered[@]}")
  if [[ "${queue[0]:-}" != "$from_variant" ]]; then
    echo "Variant ${from_variant} is already completed or not pending." >&2
    exit 2
  fi
  echo "Watching existing OAR job ${from_job} for variant ${from_variant}…"
  if ! watch_variant_job "$from_variant" "$from_job"; then
    exit 1
  fi
  queue=("${queue[@]:1}")
fi

if [[ ${#queue[@]} -eq 0 && -z "$from_job" ]]; then
  echo "All paired GNN variant reports are already completed."
  if [[ -n "$on_chain_complete" ]]; then
    echo "Running --on-chain-complete…"
    bash -lc "$on_chain_complete"
  fi
  exit 0
fi

echo "Pending paired GNN variants: ${queue[*]}"

for variant in "${queue[@]}"; do
  echo "Submitting gnn-variant ${variant}…"
  job_id="$(submit_variant "$variant")"
  if [[ -z "$job_id" || ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse OAR job id after submitting ${variant}." >&2
    exit 1
  fi
  echo "Submitted ${variant} as OAR job ${job_id}"
  if ! watch_variant_job "$variant" "$job_id"; then
    exit 1
  fi
  echo "Variant ${variant} completed."
done

echo "Paired GNN variant chain finished successfully."
if [[ -n "$on_chain_complete" ]]; then
  echo "Running --on-chain-complete…"
  bash -lc "$on_chain_complete"
fi
