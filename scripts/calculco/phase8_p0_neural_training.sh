#!/usr/bin/env bash
#OAR -n rm_phase8_p0_neural_training
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -p gpudevice<>'-1'
# Stage 2 conditional neural Transformer training (Phase 8 P0).
#
# GPU placement on Calculco/ritchie (verified 2026-07-25 on ritchie):
# - `/nodes=1/gpu=1` alone still matches CPU hosts (probe jobs landed on chimay01).
# - `#OAR -p gpudevice<>'-1'` is the inverse of CPU wrappers (`gpudevice='-1'`)
#   and forces GPU hosts (chimay31+ / visu*). Do not wrap this -p value in
#   extra double quotes inside #OAR directives: `oarsub -S` then fails with
#   "There are not enough resources" / OAR_JOB_ID=-5.
# - Optional pin: `#OAR -p network_address='chimay34' and gpudevice<>'-1'`
#   (chimay34 is dedicated=CornelIA / H100; chimay31–32 are A100).
# For a CPU-only fallback, use cores + `#OAR -p gpudevice='-1'` and set
# `NEURAL_DEVICE=cpu` in the job env.
#
# This stage is gate-first: neural work stays fail-closed until the structured
# recovery gate records neural_training_authorized=true in
# phase8_p0_gate_recovery_selection.json. PyTorch is installed on the worker via
# the optional `neural` dependency group; it is intentionally not part of the
# default environment.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

job_env="$script_dir/phase8_p0_neural_training_job.env"
if [[ -f "$job_env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$job_env"
  set +a
fi

: "${NEURAL_STAGES:=prepare train score}"
: "${NEURAL_MODE:=development}"
: "${NEURAL_SEED:=20260617}"
: "${NEURAL_TOP_K:=1,3,5,10}"
: "${NEURAL_MAX_SEQUENCE_LENGTH:=128}"
: "${NEURAL_SHARD_COUNT:=8}"
: "${DUCKDB_THREADS:=8}"
: "${DUCKDB_MEMORY_LIMIT:=24GB}"

export DUCKDB_THREADS DUCKDB_MEMORY_LIMIT
export OMP_NUM_THREADS="$DUCKDB_THREADS"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONUNBUFFERED=1

phase8_root="${PHASE8_P0_ROOT:-$DATASET_ROOT/processed/phase8_p0}"
features_root="${PHASE8_P0_FEATURES_ROOT:-$phase8_root/features}"
training_root="${PHASE8_P0_TRAINING_ROOT:-$phase8_root/training}"
neural_root="${PHASE8_P0_NEURAL_ROOT:-$phase8_root/neural}"
# Stage 2 gates against the Stage 1 recovery winner scores, not Milestone 8B.
reference_root="${PHASE8_P0_GATE_RECOVERY_ROOT:-$phase8_root/evaluation/gate_recovery}"
contract_lock="${PHASE8_P0_CONTRACT_LOCK:-$PROJECT_HOME/reports/phase8_p0_training_contract_lock.json}"
gate_selection="${PHASE8_P0_GATE_SELECTION:-$PROJECT_HOME/reports/phase8_p0_gate_recovery_selection.json}"

mkdir -p "$script_dir/logs"

echo "=== Installing the optional neural dependency group (PyTorch) ==="
uv sync --group neural

common_args=(
  --mode "$NEURAL_MODE"
  --features-root "$features_root"
  --training-root "$training_root"
  --neural-root "$neural_root"
  --reference-scores "$reference_root/baseline_scores.parquet"
  --contract-lock "$contract_lock"
  --gate-selection "$gate_selection"
  --seed "$NEURAL_SEED"
  --top-k "$NEURAL_TOP_K"
  --max-sequence-length "$NEURAL_MAX_SEQUENCE_LENGTH"
  --shard-count "$NEURAL_SHARD_COUNT"
  --duckdb-temp-dir "$DUCKDB_TEMP_DIR"
  --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT"
  --duckdb-threads "$DUCKDB_THREADS"
)
if [[ -n "${NEURAL_DEVICE:-}" ]]; then
  common_args+=(--device "$NEURAL_DEVICE")
fi
if [[ -n "${NEURAL_BATCH_RANKING_GROUPS:-}" ]]; then
  common_args+=(--batch-ranking-groups "$NEURAL_BATCH_RANKING_GROUPS")
fi
if [[ -n "${NEURAL_MAX_EPOCHS:-}" ]]; then
  common_args+=(--max-epochs "$NEURAL_MAX_EPOCHS")
fi
if [[ -n "${NEURAL_EARLY_STOPPING_PATIENCE:-}" ]]; then
  common_args+=(--early-stopping-patience "$NEURAL_EARLY_STOPPING_PATIENCE")
fi
if [[ -n "${NEURAL_LEARNING_RATE:-}" ]]; then
  common_args+=(--learning-rate "$NEURAL_LEARNING_RATE")
fi
if [[ "$NEURAL_MODE" == "final" ]]; then
  common_args+=(--frozen-selection)
fi

for stage in $NEURAL_STAGES; do
  echo "=== Phase 8 P0 neural stage: $stage ($NEURAL_MODE) ==="
  uv run python -m pipeline.neural_training "$stage" "${common_args[@]}"
done

echo "Review aggregate reports only under $PROJECT_HOME/reports:"
echo "  phase8_p0_neural_prepare_manifest.json"
echo "  phase8_p0_neural_training_evaluation.json"
echo "  phase8_p0_neural_score_evaluation.json"
echo "  phase8_p0_neural_training_selection.json"
echo "Patient-level caches, checkpoints, and scores remain under $neural_root"
