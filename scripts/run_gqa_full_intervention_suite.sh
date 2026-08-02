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
# By default every stage uses the canonical output/gqa/runs hierarchy and the
# same SUITE_TAG as its run id. Set SUITE_ROOT (or OUTPUT_ROOT as an alias) for
# the legacy/custom behavior of nesting all stages below one suite directory.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Exposed GPU configuration: use physical nvidia-smi indices.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
GPU_IDS=${GPU_IDS:-$CUDA_VISIBLE_DEVICES}
NUM_SHARDS=${NUM_SHARDS:-}
SUITE_TAG=${SUITE_TAG:-$(date +%Y%m%d_%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-${OUTPUT_ROOT:-}}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$GPU_IDS" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=0,1,... or GPU_IDS=0,1,... before starting the suite." >&2
  exit 2
fi

if [[ -n "$SUITE_ROOT" && "$DRY_RUN" != 1 ]]; then
  mkdir -p "$SUITE_ROOT"
fi
printf 'GQA full intervention suite\nRun id: %s\nCUDA_VISIBLE_DEVICES: %s\nGPU_IDS: %s\nNUM_SHARDS: %s\n' \
  "$SUITE_TAG" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$GPU_IDS" "${NUM_SHARDS:-auto}"
if [[ -n "$SUITE_ROOT" ]]; then
  echo "Custom suite root: $SUITE_ROOT"
else
  echo "Output layout: $PROJECT_ROOT/output/gqa/runs/<split>/<study>/<method>/<setting>/$SUITE_TAG"
fi

run_counterfactual_stage() {
  local perturb_mode=$1
  local perturb_position=$2
  local stage_name="${perturb_mode}/${perturb_position}_position"
  local -a output_env=("RUN_TAG=$SUITE_TAG")
  if [[ -n "$SUITE_ROOT" ]]; then
    output_env+=("OUTPUT_ROOT=$SUITE_ROOT/$stage_name")
  fi
  echo "[$(date '+%F %T')] starting stage: $stage_name"
  env -u OUTPUT_ROOT \
    GPU_IDS="$GPU_IDS" \
    NUM_SHARDS="$NUM_SHARDS" \
    PERTURB_MODE="$perturb_mode" \
    PERTURB_POSITION="$perturb_position" \
    PERTURB_INDEX= \
    DRY_RUN="$DRY_RUN" \
    "${output_env[@]}" \
    "$SCRIPT_DIR/run_gqa_counterfactual.sh"
  echo "[$(date '+%F %T')] finished stage: $stage_name"
}

run_oracle_stage() {
  local stage_name=online_oracle
  local -a output_env=("RUN_TAG=$SUITE_TAG")
  if [[ -n "$SUITE_ROOT" ]]; then
    output_env+=("OUTPUT_ROOT=$SUITE_ROOT/$stage_name")
  fi
  echo "[$(date '+%F %T')] starting stage: $stage_name"
  env -u OUTPUT_ROOT \
    GPU_IDS="$GPU_IDS" \
    NUM_SHARDS="$NUM_SHARDS" \
    DRY_RUN="$DRY_RUN" \
    "${output_env[@]}" \
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

if [[ -n "$SUITE_ROOT" ]]; then
  echo "[$(date '+%F %T')] full GQA suite finished: $SUITE_ROOT"
else
  echo "[$(date '+%F %T')] full GQA suite finished with canonical run id: $SUITE_TAG"
fi
