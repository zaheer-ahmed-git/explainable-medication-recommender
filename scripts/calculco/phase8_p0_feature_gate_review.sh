#!/usr/bin/env bash
# Write the Phase 8 P0 promotion gate review from aggregate 8B output.
# Login-node only; completes in seconds.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

phase8_evaluation_report="${PHASE8_P0_MILESTONE8B_EVAL_REPORT:-$PROJECT_HOME/reports/phase8_p0_milestone8b_ablation_evaluation.json}"
output_report="${PHASE8_P0_FEATURE_GATE_REVIEW:-$PROJECT_HOME/reports/phase8_p0_feature_gate_review.json}"

if [[ ! -f "$phase8_evaluation_report" ]]; then
  echo "ERROR: missing $phase8_evaluation_report" >&2
  echo "Run phase8_p0_graph_ablation first." >&2
  exit 1
fi

echo "phase8_evaluation_report=$phase8_evaluation_report"
echo "output_report=$output_report"

uv run python -m pipeline.feature_gate_review \
  --phase8-evaluation-report "$phase8_evaluation_report" \
  --output "$output_report"

echo "Review aggregate gate report:"
echo "  $output_report"
