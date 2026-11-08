#!/usr/bin/env bash
# Extract the first OAR job id from oarsub stdout/stderr.

set -euo pipefail

input=""
if [[ $# -gt 0 ]]; then
  input="$*"
else
  input="$(cat)"
fi

if [[ -z "$input" ]]; then
  exit 1
fi

if [[ "$input" =~ OAR_JOB_ID=([0-9]+) ]]; then
  echo "${BASH_REMATCH[1]}"
  exit 0
fi

if [[ "$input" =~ [Jj]ob[[:space:]]+[Ii][Dd][[:space:]]*[=:][[:space:]]*([0-9]+) ]]; then
  echo "${BASH_REMATCH[1]}"
  exit 0
fi

if [[ "$input" =~ [Ss]ubmitted[[:space:]].*[[:space:]]as[[:space:]]OAR[[:space:]]job[[:space:]]+([0-9]+) ]]; then
  echo "${BASH_REMATCH[1]}"
  exit 0
fi

if [[ "$input" =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
  echo "${BASH_REMATCH[1]}"
  exit 0
fi

exit 1
