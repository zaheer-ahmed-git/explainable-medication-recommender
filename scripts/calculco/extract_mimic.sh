#!/bin/bash
#OAR -n rm_mimic_extract
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

echo "=== mimic_extract start ==="
if uv run python -m pipeline.mimic_extract; then
  mimic_rc=0
else
  mimic_rc=$?
fi
echo "=== mimic_extract done exit=$mimic_rc ==="
exit "$mimic_rc"
