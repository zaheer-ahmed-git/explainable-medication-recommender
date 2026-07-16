#!/bin/bash
#OAR -n rm_phase8_p0_graph_gates
#OAR -l /nodes=1/core=8,walltime=12:00:00
#OAR -p gpudevice='-1'
# Log paths: pass at submit time, e.g.:
#   oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_gates_%jobid%.out" \
#          -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_gates_%jobid%.err" \
#          -S "$PROJECT_HOME/scripts/calculco/phase8_p0_graph_gates.sh"
#
# Chains Phase 8 P0 Milestone 8 graph suitability and Milestone 8B development
# ablation in one OAR job.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

phase8_job_env="$script_dir/phase8_p0_milestone8b_job.env"
if [[ -f "$phase8_job_env" ]]; then
  echo "=== phase8_p0_milestone8b_job.env ==="
  # shellcheck disable=SC1090
  set -a
  source "$phase8_job_env"
  set +a
  cat "$phase8_job_env"
fi

mkdir -p "$script_dir/logs"

: "${MILESTONE8B_MODE:=development}"
if [[ "$MILESTONE8B_MODE" != "development" ]]; then
  echo "phase8_p0_graph_gates.sh supports development mode only." >&2
  echo "Submit final 8B separately with submit_phase8_p0_graph_ablation.sh final." >&2
  exit 2
fi

echo "=== phase8_p0_graph_gates: milestone 8 suitability ==="
if "$script_dir/phase8_p0_graph_suitability.sh"; then
  graph_rc=0
else
  graph_rc=$?
fi
echo "=== phase8_p0_graph_gates: milestone 8 done exit=$graph_rc ==="
if (( graph_rc != 0 )); then
  exit "$graph_rc"
fi

echo "=== phase8_p0_graph_gates: milestone 8B ablation ==="
if "$script_dir/phase8_p0_graph_ablation.sh"; then
  ablation_rc=0
else
  ablation_rc=$?
fi
echo "=== phase8_p0_graph_gates: milestone 8B done exit=$ablation_rc ==="

if (( ablation_rc == 0 )); then
  echo "Next: run feature gate review on the login node with"
  echo "  scripts/calculco/phase8_p0_feature_gate_review.sh"
fi

exit "$ablation_rc"
