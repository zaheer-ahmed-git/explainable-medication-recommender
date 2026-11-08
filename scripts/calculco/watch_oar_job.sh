#!/usr/bin/env bash
# Poll one OAR job until it leaves the queue (Terminated/Error/…) and notify.
#
# Usage:
#   scripts/calculco/watch_oar_job.sh 20330
#   scripts/calculco/watch_oar_job.sh 20330 --mail you@example.com
#   OAR_WATCH_INTERVAL_SEC=120 scripts/calculco/watch_oar_job.sh 20330 \
#     --on-success 'scripts/calculco/submit_phase8_p0_gnn_training.sh development score-fusion'
#
# --on-success runs only when exit_code is 0. Does not print patient data.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

job_id="${1:-}"
if [[ -z "$job_id" || ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <oar_job_id> [--mail addr] [--on-success 'cmd'] [--on-failure 'cmd'] [--on-end 'cmd']" >&2
  exit 2
fi
shift || true

mail_to="${OAR_WATCH_MAIL:-${GNN_WATCH_MAIL:-}}"
on_success=""
on_failure=""
on_end=""
interval_sec="${OAR_WATCH_INTERVAL_SEC:-${GNN_WATCH_INTERVAL_SEC:-120}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mail)
      mail_to="${2:?--mail requires an address}"
      shift 2
      ;;
    --on-success)
      on_success="${2:?--on-success requires a command}"
      shift 2
      ;;
    --on-failure)
      on_failure="${2:?--on-failure requires a command}"
      shift 2
      ;;
    --on-end)
      on_end="${2:?--on-end requires a command}"
      shift 2
      ;;
    --interval)
      interval_sec="${2:?--interval requires seconds}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

job_field() {
  local key="$1"
  oarstat -j "$job_id" -f 2>/dev/null | awk -v k="$key" '
    $1 == k && $2 == "=" {
      sub(/^[[:space:]]*[^=]+ =[[:space:]]*/, "", $0)
      print
      exit
    }'
}

echo "Watching OAR job $job_id every ${interval_sec}s…"
while true; do
  state="$(job_field state || true)"
  if [[ -z "$state" ]]; then
    state="Gone"
    break
  fi
  case "$state" in
    Terminated|Error|Finishing)
      break
      ;;
  esac
  echo "$(date '+%F %T')  job=$job_id state=$state"
  sleep "$interval_sec"
done

for _ in 1 2 3 4 5; do
  state="$(job_field state || true)"
  [[ "$state" != "Finishing" && -n "$state" ]] && break
  sleep 2
done

exit_code="$(job_field exit_code || true)"
name="$(job_field name || true)"
host="$(job_field assigned_hostnames || true)"
stop="$(job_field stopTime || true)"
state="$(job_field state || echo Gone)"

ok=0
if [[ "$exit_code" =~ ^0[[:space:]] ]]; then
  ok=1
fi

summary=$(
  cat <<EOF
OAR job $job_id ended
  name:      ${name:-unknown}
  state:     $state
  exit_code: ${exit_code:-unknown}
  host:      ${host:-unassigned}
  stopTime:  ${stop:-unknown}
  ok:        $ok
EOF
)
echo "$summary"

if [[ -n "$mail_to" ]]; then
  subject="OAR $job_id ${name:-job} $state (ok=$ok)"
  if ! "$script_dir/send_oar_mail.sh" "$mail_to" "$subject" "$summary"; then
    echo "WARNING: notification delivery failed" >&2
  fi
fi

if [[ -n "$on_end" ]]; then
  echo "Running --on-end…"
  bash -lc "$on_end"
fi

if [[ "$ok" -eq 1 && -n "$on_success" ]]; then
  echo "Running --on-success…"
  bash -lc "$on_success"
elif [[ "$ok" -ne 1 && -n "$on_success" ]]; then
  echo "Skipping --on-success because exit_code is not 0."
fi

if [[ "$ok" -ne 1 && -n "$on_failure" ]]; then
  echo "Running --on-failure…"
  bash -lc "$on_failure"
fi

exit $((1 - ok))
