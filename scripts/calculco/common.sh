#!/usr/bin/env bash
# Shared Calculco job environment for ResearchModule pipeline CLIs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_HOME:=$HOME/ResearchModule}"

# Optional machine-specific exports (gitignored).
if [[ -f "$PROJECT_HOME/.env.calculco" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_HOME/.env.calculco"
  set +a
fi
if [[ -f "$script_dir/common.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$script_dir/common.local.sh"
fi

if [[ -z "${DATASET_ROOT:-}" ]]; then
  echo "ERROR: DATASET_ROOT is not set. Copy .env.example to .env.calculco or" >&2
  echo "create scripts/calculco/common.local.sh with exports." >&2
  exit 1
fi

: "${WORK_SCRATCH:=}"

export PROJECT_HOME DATASET_ROOT
if [[ -n "$WORK_SCRATCH" ]]; then
  export WORK_SCRATCH
fi

if [[ -n "${OAR_JOB_ID:-}" ]]; then
  job_tag="rm-job-${OAR_JOB_ID}"
else
  job_tag="rm-job-manual-$$"
fi

# Pick the first base that is actually writable. WORK_SCRATCH (e.g.
# /workdir/<lab>/<user>) is unwritable on some ritchie/chimay nodes, so prefer
# node-local /scratch before the smaller /tmp fallback.
job_scratch=""
for base in "${WORK_SCRATCH:-}" "${TMPDIR:-}" /scratch /tmp; do
  [[ -z "$base" ]] && continue
  candidate="$base/rm-scratch/$job_tag"
  if mkdir -p "$candidate/tmp" "$candidate/uv-cache" 2>/dev/null; then
    chmod 700 "$candidate" "$candidate/tmp" "$candidate/uv-cache"
    job_scratch="$candidate"
    break
  fi
done

if [[ -z "$job_scratch" ]]; then
  echo "ERROR: no writable scratch base (tried WORK_SCRATCH, TMPDIR, /scratch, /tmp)." >&2
  exit 1
fi

export TMPDIR="$job_scratch/tmp"
export UV_CACHE_DIR="$job_scratch/uv-cache"
# DuckDB does not read the OS TMPDIR; pass an explicit spill directory so
# in-memory databases can offload larger-than-memory COPY operators to disk.
export DUCKDB_TEMP_DIR="$job_scratch/tmp"

export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_HOME"

echo "PROJECT_HOME=$PROJECT_HOME"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "TMPDIR=${TMPDIR:-}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-}"
echo "DUCKDB_TEMP_DIR=${DUCKDB_TEMP_DIR:-}"
echo "Host=$(hostname) UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
