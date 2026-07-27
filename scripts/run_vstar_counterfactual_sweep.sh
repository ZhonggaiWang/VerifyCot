#!/usr/bin/env bash
# Launch VStar counterfactual runs whenever a sufficiently idle GPU is found.
# Usage: scripts/run_vstar_counterfactual_sweep.sh {random|first|last|all}
# Set INTERVENTION_MODE=random_box (default) or remove_grounding.

set -euo pipefail

MODE=${1:-}
case "$MODE" in
  random|first|last|all) ;;
  *)
    echo "Usage: $0 {random|first|last|all}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNS=${RUNS:-2}
GPU_IDLE_MEMORY_MB=${GPU_IDLE_MEMORY_MB:-500}
POLL_SECONDS=${POLL_SECONDS:-30}
# Set GPU_IDS=0,1,2 (or CUDA_VISIBLE_DEVICES=0,1,2) before launching to
# reserve every GPU not listed here.  Values are physical nvidia-smi indices.
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
VSTAR_QUESTIONS_PATH=${VSTAR_QUESTIONS_PATH:-/data/zhonggai/VStar/test_questions.jsonl}
VSTAR_IMAGE_DIR=${VSTAR_IMAGE_DIR:-/data/zhonggai/VStar}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
IOU_MAX=${IOU_MAX:-0.1}
INTERVENTION_MODE=${INTERVENTION_MODE:-random_box}
case "$INTERVENTION_MODE" in
  random_box|remove_grounding) ;;
  *)
    echo "INTERVENTION_MODE must be random_box or remove_grounding, got: $INTERVENTION_MODE" >&2
    exit 2
    ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/output/vstar/counterfactual}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="$OUTPUT_ROOT/$INTERVENTION_MODE/$RUN_TAG"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to detect idle GPUs." >&2
  exit 1
fi
if [[ ! -f "$VSTAR_QUESTIONS_PATH" ]]; then
  echo "VStar questions file not found: $VSTAR_QUESTIONS_PATH" >&2
  exit 1
fi
if [[ ! -d "$VSTAR_IMAGE_DIR" ]]; then
  echo "VStar image directory not found: $VSTAR_IMAGE_DIR" >&2
  exit 1
fi

declare -a TASK_GROUPS=()
declare -a TASK_POSITIONS=()
declare -a TASK_RUNS=()

add_group() {
  local group=$1
  local position=$2
  local run
  for ((run = 1; run <= RUNS; run++)); do
    TASK_GROUPS+=("$group")
    TASK_POSITIONS+=("$position")
    TASK_RUNS+=("$run")
  done
}

case "$MODE" in
  random) add_group random_position random ;;
  first)  add_group first_position first ;;
  last)   add_group last_position last ;;
  all)
    add_group random_position random
    add_group first_position first
    add_group last_position last
    ;;
esac

mkdir -p "$RUN_ROOT"
echo "Intervention mode: $INTERVENTION_MODE"
echo "Run root: $RUN_ROOT"
echo "Tasks: ${#TASK_GROUPS[@]}; idle GPU threshold: ${GPU_IDLE_MEMORY_MB} MB"

declare -A ACTIVE_PID_BY_GPU=()
declare -A ACTIVE_LABEL_BY_GPU=()
declare -A ALLOWED_GPUS=()
if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -r -a requested_gpus <<< "$GPU_IDS"
  for gpu in "${requested_gpus[@]}"; do
    gpu=${gpu//[[:space:]]/}
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "GPU_IDS/CUDA_VISIBLE_DEVICES must be comma-separated physical GPU indices: $GPU_IDS" >&2
      exit 2
    fi
    ALLOWED_GPUS[$gpu]=1
  done
  echo "Allowed physical GPUs: $GPU_IDS"
else
  echo "Allowed physical GPUs: all"
fi
next_task=0

idle_gpus() {
  local gpu used
  while IFS=, read -r gpu used; do
    gpu=${gpu//[[:space:]]/}
    used=${used//[[:space:]]/}
    [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
    if (( ${#ALLOWED_GPUS[@]} > 0 )) && [[ -z ${ALLOWED_GPUS[$gpu]+x} ]]; then
      continue
    fi
    if (( used <= GPU_IDLE_MEMORY_MB )); then
      printf '%s\n' "$gpu"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

reap_finished() {
  local gpu pid label status
  for gpu in "${!ACTIVE_PID_BY_GPU[@]}"; do
    pid=${ACTIVE_PID_BY_GPU[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      label=${ACTIVE_LABEL_BY_GPU[$gpu]}
      status=0
      wait "$pid" || status=$?
      echo "[$(date '+%F %T')] finished gpu=$gpu task=$label status=$status"
      unset 'ACTIVE_PID_BY_GPU[$gpu]'
      unset 'ACTIVE_LABEL_BY_GPU[$gpu]'
    fi
  done
}

launch_task() {
  local gpu=$1
  local group=${TASK_GROUPS[$next_task]}
  local position=${TASK_POSITIONS[$next_task]}
  local run_number=${TASK_RUNS[$next_task]}
  local task_dir="$RUN_ROOT/$group/run_$(printf '%02d' "$run_number")"
  local output_path="$task_dir/results.jsonl"
  local log_path="$task_dir/run.log"
  local label="$group/run_$(printf '%02d' "$run_number")"

  mkdir -p "$task_dir"
  echo "[$(date '+%F %T')] launching gpu=$gpu task=$label"
  (
    cd "$PROJECT_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 conda run -n vocot python -u \
      eval/Oracle_experiment/vstar/evaluate_counterfactual.py \
      --model-path "$MODEL_PATH" \
      --questions-path "$VSTAR_QUESTIONS_PATH" \
      --image-dir "$VSTAR_IMAGE_DIR" \
      --output "$output_path" \
      --perturb-position "$position" \
      --perturb-mode "$INTERVENTION_MODE" \
      --random-seeds \
      --iou-max "$IOU_MAX" \
      --max-new-tokens "$MAX_NEW_TOKENS"
  ) >"$log_path" 2>&1 &
  ACTIVE_PID_BY_GPU[$gpu]=$!
  ACTIVE_LABEL_BY_GPU[$gpu]=$label
  ((next_task += 1))
}

cleanup() {
  local gpu
  echo "Stopping launched jobs..." >&2
  for gpu in "${!ACTIVE_PID_BY_GPU[@]}"; do
    kill "${ACTIVE_PID_BY_GPU[$gpu]}" 2>/dev/null || true
  done
  exit 130
}
trap cleanup INT TERM

while (( next_task < ${#TASK_GROUPS[@]} || ${#ACTIVE_PID_BY_GPU[@]} > 0 )); do
  reap_finished
  if (( next_task < ${#TASK_GROUPS[@]} )); then
    while IFS= read -r gpu; do
      (( next_task < ${#TASK_GROUPS[@]} )) || break
      [[ -n ${ACTIVE_PID_BY_GPU[$gpu]+x} ]] && continue
      launch_task "$gpu"
    done < <(idle_gpus)
  fi
  if (( next_task < ${#TASK_GROUPS[@]} || ${#ACTIVE_PID_BY_GPU[@]} > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date '+%F %T')] all tasks finished: $RUN_ROOT"
