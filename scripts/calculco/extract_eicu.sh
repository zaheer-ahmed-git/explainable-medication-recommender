#!/bin/bash
#OAR -n rm_eicu_extract
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time (see extract_mimic.sh header).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

echo "=== eicu_extract start ==="
if uv run python -m pipeline.eicu_extract; then
  eicu_rc=0
else
  eicu_rc=$?
fi
echo "=== eicu_extract done exit=$eicu_rc ==="
exit "$eicu_rc"
