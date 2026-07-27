#!/usr/bin/env bash
# Run the complete GQA grounding-intervention suite sequentially.
#
# Stages (one stage fully finishes and merges before the next begins):
#   1-3 remove_grounding at random / first / last coordinates
#   4-6 random_box at random / first / last coordinates
#   7   online GT-grounding oracle
#
# Choose physical GPUs before launch, for example:
#   CUDA_VISIBLE_DEVICES=0,1,2 NUM_SHARDS=3 scripts/run_gqa_full_intervention_suite.sh
# GPU_IDS takes precedence over CUDA_VISIBLE_DEVICES when both are set.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Exposed GPU configuration: use physical nvidia-smi indices.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
GPU_IDS=${GPU_IDS:-$CUDA_VISIBLE_DEVICES}
NUM_SHARDS=${NUM_SHARDS:-}
SUITE_TAG=${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-$PROJECT_ROOT/output/gqa/full_intervention_suite/$SUITE_TAG}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$GPU_IDS" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=0,1,... or GPU_IDS=0,1,... before starting the suite." >&2
  exit 2
fi

mkdir -p "$SUITE_ROOT"
printf 'GQA full intervention suite\nSuite root: %s\nCUDA_VISIBLE_DEVICES: %s\nGPU_IDS: %s\nNUM_SHARDS: %s\n' \
  "$SUITE_ROOT" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$GPU_IDS" "${NUM_SHARDS:-auto}"

run_counterfactual_stage() {
  local perturb_mode=$1
  local perturb_position=$2
  local stage_name="${perturb_mode}/${perturb_position}_position"
  echo "[$(date '+%F %T')] starting stage: $stage_name"
  env \
    GPU_IDS="$GPU_IDS" \
    NUM_SHARDS="$NUM_SHARDS" \
    OUTPUT_ROOT="$SUITE_ROOT/$stage_name" \
    PERTURB_MODE="$perturb_mode" \
    PERTURB_POSITION="$perturb_position" \
    PERTURB_INDEX= \
    DRY_RUN="$DRY_RUN" \
    "$SCRIPT_DIR/run_gqa_counterfactual.sh"
  echo "[$(date '+%F %T')] finished stage: $stage_name"
}

run_oracle_stage() {
  local stage_name=online_oracle
  echo "[$(date '+%F %T')] starting stage: $stage_name"
  env \
    GPU_IDS="$GPU_IDS" \
    NUM_SHARDS="$NUM_SHARDS" \
    OUTPUT_ROOT="$SUITE_ROOT/$stage_name" \
    DRY_RUN="$DRY_RUN" \
    "$SCRIPT_DIR/run_gqa_online_oracle.sh"
  echo "[$(date '+%F %T')] finished stage: $stage_name"
}

# Do not parallelise these calls: their order is the experimental schedule.
run_counterfactual_stage remove_grounding random
run_counterfactual_stage remove_grounding first
run_counterfactual_stage remove_grounding last
run_counterfactual_stage random_box random
run_counterfactual_stage random_box first
run_counterfactual_stage random_box last
run_oracle_stage

echo "[$(date '+%F %T')] full GQA suite finished: $SUITE_ROOT"
