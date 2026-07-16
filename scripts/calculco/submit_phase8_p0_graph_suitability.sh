#!/usr/bin/env bash
# Submit Phase 8 P0 Milestone 8 graph suitability through OAR.
#
# Usage:
#   scripts/calculco/submit_phase8_p0_graph_suitability.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_HOME:=$HOME/ResearchModule}"

mkdir -p "$script_dir/logs"

oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_phase8_p0_graph_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/phase8_p0_graph_suitability.sh"
