#!/bin/bash
#OAR -n rm_profile
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
# CPU-only jobs should use: #OAR -p gpudevice='-1'
# During the 2026 Calculco migration, legacy CPU nodes on the calculco front-end
# may be offline while GPU nodes remain Alive. If submission fails with "not
# enough resources", SSH to ritchie.univ-littoral.fr (chimay CPU nodes) and
# uncomment the gpudevice line above. Do not prefix disabled directives with #OAR;
# oarsub parses every #OAR line in this file.
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/profile_tables.sh"
#
# Re-runs the FULL aggregate quality profile over every configured source
# table and overwrites reports/quality_profile.json. A full run is required
# (not a --table subset) because pipeline.profile_tables rewrites the whole
# report; a partial run would drop the other tables' entries that the
# extraction gates depend on. Use this after correcting local chartevents /
# inputevents source files so their scan_failed entries refresh.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

mkdir -p "$script_dir/logs"

echo "=== profile_tables preflight ==="
preflight_fail=0
for rel in \
  "mimiciv/3.1/icu/chartevents.csv.gz" \
  "mimiciv/3.1/icu/inputevents.csv.gz"; do
  if [[ -f "$DATASET_ROOT/$rel" ]]; then
    echo "preflight ok: $rel"
  else
    echo "preflight MISSING: $rel ($DATASET_ROOT/$rel)" >&2
    preflight_fail=1
  fi
done
if (( preflight_fail != 0 )); then
  echo "Preflight failed; corrected source files are required before profiling." >&2
  exit 1
fi

echo "=== profile_tables start (full re-profile) ==="
if uv run python -m pipeline.profile_tables; then
  profile_rc=0
else
  profile_rc=$?
fi
echo "=== profile_tables done exit=$profile_rc ==="

if (( profile_rc == 0 )); then
  echo "Review aggregate report under $PROJECT_HOME/reports/:"
  echo "  quality_profile.json (confirm mimic_chartevents and mimic_inputevents are 'completed')"
  echo "Then re-run source extraction so inputevents materializes past its gate."
fi

exit "$profile_rc"
