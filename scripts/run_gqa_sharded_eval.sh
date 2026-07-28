#!/usr/bin/env bash
# Run GQA counterfactual or online-oracle evaluation in non-overlapping shards.
# Usage: scripts/run_gqa_sharded_eval.sh {counterfactual|oracle}

set -euo pipefail

EXPERIMENT=${1:-}
case "$EXPERIMENT" in
  counterfactual|oracle) ;;
  *)
    echo "Usage: $0 {counterfactual|oracle}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
MANIFEST_PATH=${MANIFEST_PATH:-$PROJECT_ROOT/output/gqa/annotations/oracle_val_1000/manifest.jsonl}
MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
FINAL_MAX_NEW_TOKENS=${FINAL_MAX_NEW_TOKENS:-32}
TEMPERATURE=${TEMPERATURE:-0.0}
GPU_IDLE_MEMORY_MB=${GPU_IDLE_MEMORY_MB:-500}
POLL_SECONDS=${POLL_SECONDS:-30}
# Physical GPU indices.  Set this explicitly to reserve all other GPUs.
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
START_INDEX=${START_INDEX:-0}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-}
NUM_SHARDS=${NUM_SHARDS:-}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
NO_RESUME=${NO_RESUME:-0}
DRY_RUN=${DRY_RUN:-0}

# Counterfactual-only controls.  PERTURB_INDEX is a 1-based fixed coordinate
# index and is mutually exclusive with a non-random PERTURB_POSITION.
PERTURB_MODE=${PERTURB_MODE:-random_box}
PERTURB_POSITION=${PERTURB_POSITION:-random}
PERTURB_INDEX=${PERTURB_INDEX:-}
SELECTION_SEED=${SELECTION_SEED:-2026}
PERTURB_SEED=${PERTURB_SEED:-2027}
IOU_MIN=${IOU_MIN:-0.0}
IOU_MAX=${IOU_MAX:-0.1}
PERTURB_BOX_MODE=${PERTURB_BOX_MODE:-random}
RANDOM_BOX_MIN_SIZE=${RANDOM_BOX_MIN_SIZE:-0.05}
RANDOM_BOX_MAX_SIZE=${RANDOM_BOX_MAX_SIZE:-0.5}
CONTEXT_WINDOW_TOKENS=${CONTEXT_WINDOW_TOKENS:-48}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to detect idle GPUs." >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "GQA manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi
if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "START_INDEX must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$EVAL_MAX_SAMPLES" && ! "$EVAL_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_SAMPLES must be a positive integer when set." >&2
  exit 2
fi

declare -A ALLOWED_GPUS=()
declare -a GPU_LIST=()
if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -r -a requested_gpus <<< "$GPU_IDS"
  for gpu in "${requested_gpus[@]}"; do
    gpu=${gpu//[[:space:]]/}
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "GPU_IDS/CUDA_VISIBLE_DEVICES must be comma-separated physical GPU indices." >&2
      exit 2
    fi
    if [[ -z ${ALLOWED_GPUS[$gpu]+x} ]]; then
      ALLOWED_GPUS[$gpu]=1
      GPU_LIST+=("$gpu")
    fi
  done
else
  while IFS= read -r gpu; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || continue
    ALLOWED_GPUS[$gpu]=1
    GPU_LIST+=("$gpu")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "No usable GPU indices were found." >&2
  exit 1
fi

if [[ -z "$NUM_SHARDS" ]]; then
  NUM_SHARDS=${#GPU_LIST[@]}
elif [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer." >&2
  exit 2
fi

MANIFEST_COUNT=$(wc -l < "$MANIFEST_PATH")
if (( START_INDEX >= MANIFEST_COUNT )); then
  echo "START_INDEX=$START_INDEX is outside manifest size $MANIFEST_COUNT." >&2
  exit 2
fi
AVAILABLE_SAMPLES=$(( MANIFEST_COUNT - START_INDEX ))
if [[ -z "$EVAL_MAX_SAMPLES" ]]; then
  TOTAL_SAMPLES=$AVAILABLE_SAMPLES
else
  TOTAL_SAMPLES=$EVAL_MAX_SAMPLES
  if (( TOTAL_SAMPLES > AVAILABLE_SAMPLES )); then
    TOTAL_SAMPLES=$AVAILABLE_SAMPLES
  fi
fi
if (( NUM_SHARDS > TOTAL_SAMPLES )); then
  NUM_SHARDS=$TOTAL_SAMPLES
fi

if [[ "$EXPERIMENT" == counterfactual ]]; then
  case "$PERTURB_MODE" in random_box|remove_grounding) ;; *)
    echo "PERTURB_MODE must be random_box or remove_grounding." >&2; exit 2;; esac
  case "$PERTURB_POSITION" in random|first|last) ;; *)
    echo "PERTURB_POSITION must be random, first, or last." >&2; exit 2;; esac
  if [[ -n "$PERTURB_INDEX" ]]; then
    if [[ ! "$PERTURB_INDEX" =~ ^[1-9][0-9]*$ ]]; then
      echo "PERTURB_INDEX must be a positive 1-based index." >&2
      exit 2
    fi
    if [[ "$PERTURB_POSITION" != random ]]; then
      echo "PERTURB_INDEX cannot be combined with PERTURB_POSITION=$PERTURB_POSITION." >&2
      exit 2
    fi
    POSITION_LABEL="index_${PERTURB_INDEX}"
  else
    POSITION_LABEL="${PERTURB_POSITION}_position"
  fi
  RUN_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/output/gqa/counterfactual/$PERTURB_MODE/$POSITION_LABEL/$RUN_TAG}
else
  RUN_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/output/gqa/online_oracle/$RUN_TAG}
fi

declare -a TASK_STARTS=()
declare -a TASK_COUNTS=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  shard_start=$(( START_INDEX + (TOTAL_SAMPLES * shard) / NUM_SHARDS ))
  shard_end=$(( START_INDEX + (TOTAL_SAMPLES * (shard + 1)) / NUM_SHARDS ))
  TASK_STARTS+=("$shard_start")
  TASK_COUNTS+=("$(( shard_end - shard_start ))")
done

mkdir -p "$RUN_ROOT/shards"
printf 'Experiment: %s\nRun root: %s\nManifest: %s\nSamples: [%s, %s), total=%s; shards=%s\nAllowed physical GPUs: %s\n' \
  "$EXPERIMENT" "$RUN_ROOT" "$MANIFEST_PATH" "$START_INDEX" "$(( START_INDEX + TOTAL_SAMPLES ))" \
  "$TOTAL_SAMPLES" "$NUM_SHARDS" "${GPU_LIST[*]}"
if [[ "$EXPERIMENT" == counterfactual ]]; then
  printf 'Counterfactual: mode=%s, position=%s, index=%s\n' \
    "$PERTURB_MODE" "$PERTURB_POSITION" "${PERTURB_INDEX:-none}"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    printf 'shard_%03d: start=%s count=%s\n' "$shard" "${TASK_STARTS[$shard]}" "${TASK_COUNTS[$shard]}"
  done
  exit 0
fi

declare -A ACTIVE_PID_BY_GPU=()
declare -A ACTIVE_LABEL_BY_GPU=()
next_task=0
FAILED_TASKS=0

idle_gpus() {
  local gpu used
  while IFS=, read -r gpu used; do
    gpu=${gpu//[[:space:]]/}
    used=${used//[[:space:]]/}
    [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
    [[ -n ${ALLOWED_GPUS[$gpu]+x} ]] || continue
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
      if (( status != 0 )); then
        FAILED_TASKS=$(( FAILED_TASKS + 1 ))
      fi
      unset 'ACTIVE_PID_BY_GPU[$gpu]'
      unset 'ACTIVE_LABEL_BY_GPU[$gpu]'
    fi
  done
}

launch_task() {
  local gpu=$1
  local shard=$next_task
  local start=${TASK_STARTS[$shard]}
  local count=${TASK_COUNTS[$shard]}
  local task_dir="$RUN_ROOT/shards/shard_$(printf '%03d' "$shard")"
  local output_path="$task_dir/results.jsonl"
  local log_path="$task_dir/run.log"
  local -a extra_args=()
  mkdir -p "$task_dir"
  if [[ "$NO_RESUME" == 1 ]]; then
    extra_args+=(--no-resume)
  fi
  if [[ "$EXPERIMENT" == counterfactual ]]; then
    extra_args+=(
      --perturb-mode "$PERTURB_MODE"
      --selection-seed "$SELECTION_SEED"
      --perturb-seed "$PERTURB_SEED"
      --iou-min "$IOU_MIN" --iou-max "$IOU_MAX"
      --perturb-box-mode "$PERTURB_BOX_MODE"
      --random-box-min-size "$RANDOM_BOX_MIN_SIZE"
      --random-box-max-size "$RANDOM_BOX_MAX_SIZE"
    )
    if [[ -n "$PERTURB_INDEX" ]]; then
      extra_args+=(--perturb-index "$PERTURB_INDEX")
    else
      extra_args+=(--perturb-position "$PERTURB_POSITION")
    fi
    eval_script=eval/Oracle_experiment/gqa/evaluate_counterfactual.py
  else
    extra_args+=(--context-window-tokens "$CONTEXT_WINDOW_TOKENS")
    eval_script=eval/Oracle_experiment/gqa/evaluate_online_oracle.py
  fi
  echo "[$(date '+%F %T')] launching gpu=$gpu shard=$shard start=$start count=$count"
  (
    cd "$PROJECT_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 conda run -n vocot python -u "$eval_script" \
      --model-path "$MODEL_PATH" --manifest-path "$MANIFEST_PATH" --output "$output_path" \
      --start-index "$start" --max-samples "$count" \
      --max-new-tokens "$MAX_NEW_TOKENS" --final-max-new-tokens "$FINAL_MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" "${extra_args[@]}"
  ) >"$log_path" 2>&1 &
  ACTIVE_PID_BY_GPU[$gpu]=$!
  ACTIVE_LABEL_BY_GPU[$gpu]="shard_$(printf '%03d' "$shard")"
  next_task=$(( next_task + 1 ))
}

cleanup() {
  local gpu
  echo "Stopping launched GQA jobs..." >&2
  for gpu in "${!ACTIVE_PID_BY_GPU[@]}"; do
    kill "${ACTIVE_PID_BY_GPU[$gpu]}" 2>/dev/null || true
  done
  exit 130
}
trap cleanup INT TERM

while (( next_task < NUM_SHARDS || ${#ACTIVE_PID_BY_GPU[@]} > 0 )); do
  reap_finished
  if (( next_task < NUM_SHARDS )); then
    while IFS= read -r gpu; do
      (( next_task < NUM_SHARDS )) || break
      [[ -n ${ACTIVE_PID_BY_GPU[$gpu]+x} ]] && continue
      launch_task "$gpu"
    done < <(idle_gpus)
  fi
  if (( next_task < NUM_SHARDS || ${#ACTIVE_PID_BY_GPU[@]} > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done

if (( FAILED_TASKS > 0 )); then
  echo "$FAILED_TASKS shard job(s) failed; inspect $RUN_ROOT/shards/*/run.log." >&2
  exit 1
fi

if [[ "$EXPERIMENT" == counterfactual ]]; then
  settings=$(printf '{"dataset":"gqa_val_manifest","sharded":true,"perturb_mode":"%s","perturb_position":"%s","perturb_index":%s,"selection_seed":%s,"perturb_seed":%s,"iou_range":[%s,%s],"perturb_box_mode":"%s","max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s}' \
    "$PERTURB_MODE" "$PERTURB_POSITION" "${PERTURB_INDEX:-null}" "$SELECTION_SEED" "$PERTURB_SEED" \
    "$IOU_MIN" "$IOU_MAX" "$PERTURB_BOX_MODE" "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE")
else
  settings=$(printf '{"dataset":"gqa_val_manifest","sharded":true,"oracle_mode":"online_explicit_target_oracle","oracle_box_coordinate_system":"normalized_xyxy_on_center_padded_square","context_window_tokens":%s,"max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s}' \
    "$CONTEXT_WINDOW_TOKENS" "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE")
fi
cd "$PROJECT_ROOT"
conda run -n vocot python eval/Oracle_experiment/gqa/aggregate_results.py \
  --experiment "$EXPERIMENT" --shards-dir "$RUN_ROOT/shards" \
  --output "$RUN_ROOT/results.jsonl" --settings "$settings"
echo "[$(date '+%F %T')] all GQA shards finished: $RUN_ROOT"
