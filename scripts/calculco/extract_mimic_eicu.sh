#!/bin/bash
#OAR -n rm_extract
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time (see extract_mimic.sh header).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

mimic_rc=0
eicu_rc=0

echo "=== mimic_extract start ==="
if uv run python -m pipeline.mimic_extract; then
  mimic_rc=0
else
  mimic_rc=$?
fi
echo "=== mimic_extract done exit=$mimic_rc ==="

echo "=== eicu_extract start ==="
if uv run python -m pipeline.eicu_extract; then
  eicu_rc=0
else
  eicu_rc=$?
fi
echo "=== eicu_extract done exit=$eicu_rc ==="

if (( mimic_rc != 0 || eicu_rc != 0 )); then
  echo "One or more extractors failed: mimic_rc=$mimic_rc eicu_rc=$eicu_rc"
  exit 1
fi
