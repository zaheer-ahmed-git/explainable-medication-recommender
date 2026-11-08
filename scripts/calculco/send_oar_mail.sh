#!/usr/bin/env bash
# Best-effort login-node mail helper for OAR watchers.
#
# Calculco hands messages to opale-mta.univ-littoral.fr; Gmail often delays or
# filters them. This script always writes a local log and optionally pushes to
# ntfy.sh when OAR_WATCH_NTFY_TOPIC is set.
#
# Usage:
#   scripts/calculco/send_oar_mail.sh you@example.com "subject" "body line 1"
#   OAR_WATCH_NTFY_TOPIC=my-secret-topic scripts/calculco/send_oar_mail.sh ...

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <address> <subject> [body lines…]" >&2
  exit 2
fi

mail_to="$1"
subject="$2"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_home="${PROJECT_HOME:-$(cd "$script_dir/../.." && pwd)}"
log_dir="$project_home/scripts/calculco/logs"
mkdir -p "$log_dir"

body_file="$(mktemp "${TMPDIR:-/tmp}/rm_oar_mail.XXXXXX")"
verbose_file="$(mktemp "${TMPDIR:-/tmp}/rm_oar_mail_verbose.XXXXXX")"
cleanup() {
  rm -f "$body_file" "$verbose_file"
}
trap cleanup EXIT

{
  printf '%s\n' "$@"
} >"$body_file"

from_addr="${OAR_WATCH_MAIL_FROM:-${USER}@calculco.univ-littoral.fr}"
timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
stamp_file="$(date -u '+%Y%m%dT%H%M%SZ')"
log_path="$log_dir/oar_notify_${stamp_file}.txt"
latest_path="$log_dir/oar_notify_latest.txt"

write_local_log() {
  local mail_status="$1"
  {
    printf 'timestamp: %s\n' "$timestamp"
    printf 'to: %s\n' "$mail_to"
    printf 'from: %s\n' "$from_addr"
    printf 'subject: %s\n' "$subject"
    printf 'mail_status: %s\n\n' "$mail_status"
    cat "$body_file"
    if [[ -s "$verbose_file" ]]; then
      printf '\n--- mail verbose ---\n'
      cat "$verbose_file"
    fi
  } >"$log_path"
  cp "$log_path" "$latest_path"
  echo "Local notification log: $log_path"
}

push_ntfy() {
  local topic="${OAR_WATCH_NTFY_TOPIC:-}"
  if [[ -z "$topic" ]]; then
    return 0
  fi
  local server="${OAR_WATCH_NTFY_SERVER:-https://ntfy.sh}"
  local body
  body="$(cat "$body_file")"
  if curl -fsS \
    -H "Title: $subject" \
    -H "Priority: high" \
    -d "$body" \
    "${server%/}/${topic}" >/dev/null 2>&1; then
    echo "Pushed notification to ntfy topic ${topic}"
    return 0
  fi
  echo "WARNING: ntfy push to ${topic} failed" >&2
  return 1
}

mail_status="not_attempted"
mail_ok=0
if command -v mail >/dev/null 2>&1; then
  if mail -v -r "$from_addr" -s "$subject" "$mail_to" <"$body_file" >"$verbose_file" 2>&1; then
    if grep -q 'queued as' "$verbose_file" || grep -q '250 2.0.0 Ok' "$verbose_file"; then
      mail_status="queued_on_opale_mta"
      mail_ok=1
      echo "Mailed notification to $mail_to (accepted by university relay)"
    else
      mail_status="mail_returned_zero_without_queue_confirm"
    fi
  else
    mail_status="mail_command_failed"
  fi
elif command -v mailx >/dev/null 2>&1; then
  if mailx -v -r "$from_addr" -s "$subject" "$mail_to" <"$body_file" >"$verbose_file" 2>&1; then
    mail_status="mailx_sent"
    mail_ok=1
    echo "Mailed notification to $mail_to via mailx"
  else
    mail_status="mailx_command_failed"
  fi
else
  mail_status="no_mail_binary"
fi

write_local_log "$mail_status"
push_ntfy || true

if [[ "$mail_ok" -eq 1 ]]; then
  exit 0
fi

echo "WARNING: SMTP handoff did not confirm queue acceptance; see $log_path" >&2
exit 1
