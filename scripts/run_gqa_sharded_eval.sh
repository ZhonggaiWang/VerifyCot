#!/usr/bin/env bash
# Run GQA evaluations directly or in non-overlapping shards.
# Usage: scripts/run_gqa_sharded_eval.sh {counterfactual|oracle|selective_router|testdev_baseline}
#
# New runs default to:
#   output/gqa/runs/<split>/<study>/<method>/<setting>/<run_id>/
# OUTPUT_ROOT remains a backwards-compatible override for the complete run
# directory. RUN_SPLIT overrides the canonical split component when a custom
# MANIFEST_PATH is used.

set -euo pipefail

EXPERIMENT=${1:-}
case "$EXPERIMENT" in
  counterfactual|oracle|selective_router|testdev_baseline) ;;
  *)
    echo "Usage: $0 {counterfactual|oracle|selective_router|testdev_baseline}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [[ "$EXPERIMENT" == testdev_baseline ]]; then
  DEFAULT_MANIFEST_PATH=$PROJECT_ROOT/output/gqa/annotations/testdev_balanced/manifest.jsonl
else
  DEFAULT_MANIFEST_PATH=$PROJECT_ROOT/output/gqa/annotations/oracle_val_1000/manifest.jsonl
fi
MANIFEST_PATH=${MANIFEST_PATH:-$DEFAULT_MANIFEST_PATH}
MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
VOCOT_PYTHON=${VOCOT_PYTHON:-/home/zhonggai/miniconda3/envs/vocot/bin/python}
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
RUN_SPLIT=${RUN_SPLIT:-}
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
# Selective-router-only controls.  BASELINE_RESULTS must be the padding-fixed
# paired run because it supplies both untouched baseline answers and GT boxes.
BASELINE_RESULTS=${BASELINE_RESULTS:-$PROJECT_ROOT/output/gqa/online_oracle/padding_fix_v1/results.jsonl}
IOU_THRESHOLD=${IOU_THRESHOLD:-0.1}
MODEL_LOAD_LOCK_TIMEOUT_SECONDS=${MODEL_LOAD_LOCK_TIMEOUT_SECONDS:-300}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to detect idle GPUs." >&2
  exit 1
fi
if [[ ! -x "$VOCOT_PYTHON" ]]; then
  echo "vocot Python executable not found: $VOCOT_PYTHON" >&2
  echo "Set VOCOT_PYTHON to the Python executable inside the vocot environment." >&2
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
  SETTING_NAME=${POSITION_LABEL%_position}
  SPLIT_NAME=${RUN_SPLIT:-val_1000_dev}
  DEFAULT_RUN_ROOT=$PROJECT_ROOT/output/gqa/runs/$SPLIT_NAME/counterfactual/$PERTURB_MODE/$SETTING_NAME/$RUN_TAG
elif [[ "$EXPERIMENT" == oracle ]]; then
  SPLIT_NAME=${RUN_SPLIT:-val_1000_dev}
  DEFAULT_RUN_ROOT=$PROJECT_ROOT/output/gqa/runs/$SPLIT_NAME/oracle/always_gt/default/$RUN_TAG
elif [[ "$EXPERIMENT" == selective_router ]]; then
  if [[ ! -f "$BASELINE_RESULTS" ]]; then
    echo "Padding-fixed GQA baseline results not found: $BASELINE_RESULTS" >&2
    exit 1
  fi
  SPLIT_NAME=${RUN_SPLIT:-val_1000_dev}
  IOU_SETTING=${IOU_THRESHOLD//./p}
  DEFAULT_RUN_ROOT=$PROJECT_ROOT/output/gqa/runs/$SPLIT_NAME/routing/oracle_verifier__oracle_experts/iou_$IOU_SETTING/$RUN_TAG
else
  SPLIT_NAME=${RUN_SPLIT:-testdev_12578}
  DEFAULT_RUN_ROOT=$PROJECT_ROOT/output/gqa/runs/$SPLIT_NAME/baseline/volcano_7b/default/$RUN_TAG
fi
RUN_ROOT=${OUTPUT_ROOT:-$DEFAULT_RUN_ROOT}
if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$PROJECT_ROOT/$RUN_ROOT"
fi

declare -a TASK_STARTS=()
declare -a TASK_COUNTS=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  shard_start=$(( START_INDEX + (TOTAL_SAMPLES * shard) / NUM_SHARDS ))
  shard_end=$(( START_INDEX + (TOTAL_SAMPLES * (shard + 1)) / NUM_SHARDS ))
  TASK_STARTS+=("$shard_start")
  TASK_COUNTS+=("$(( shard_end - shard_start ))")
done

USE_SHARDS=0
if (( NUM_SHARDS > 1 )); then
  USE_SHARDS=1
fi

printf 'Experiment: %s\nRun root: %s\nManifest: %s\nSamples: [%s, %s), total=%s; partitions=%s\nAllowed physical GPUs: %s\n' \
  "$EXPERIMENT" "$RUN_ROOT" "$MANIFEST_PATH" "$START_INDEX" "$(( START_INDEX + TOTAL_SAMPLES ))" \
  "$TOTAL_SAMPLES" "$NUM_SHARDS" "${GPU_LIST[*]}"
if (( USE_SHARDS )); then
  echo 'Execution layout: sharded (partial results under run root/shards; merged results at run root)'
else
  echo 'Execution layout: direct (results written at run root; no shards directory)'
fi
if [[ "$EXPERIMENT" == counterfactual ]]; then
  printf 'Counterfactual: mode=%s, position=%s, index=%s\n' \
    "$PERTURB_MODE" "$PERTURB_POSITION" "${PERTURB_INDEX:-none}"
elif [[ "$EXPERIMENT" == selective_router ]]; then
  printf 'Selective router: IoU threshold=%s; baseline=%s\n' \
    "$IOU_THRESHOLD" "$BASELINE_RESULTS"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    if (( USE_SHARDS )); then
      printf 'shard_%03d: start=%s count=%s output=%s/shards/shard_%03d/results.jsonl\n' \
        "$shard" "${TASK_STARTS[$shard]}" "${TASK_COUNTS[$shard]}" "$RUN_ROOT" "$shard"
    else
      printf 'run: start=%s count=%s output=%s/results.jsonl\n' \
        "${TASK_STARTS[$shard]}" "${TASK_COUNTS[$shard]}" "$RUN_ROOT"
    fi
  done
  exit 0
fi

mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_ROOT/artifacts"

# Resuming is safe only when the data partition and semantic setting are
# identical.  In particular, changing NUM_SHARDS under the same RUN_TAG can
# leave stale shard directories that would otherwise be merged silently.
MANIFEST_ABS=$(realpath "$MANIFEST_PATH")
PARTITION_PLAN_PATH="$RUN_ROOT/artifacts/partition_plan.txt"
PARTITION_PLAN=$(printf '%s\n' \
  'schema_version=1' \
  "experiment=$EXPERIMENT" \
  "split=$SPLIT_NAME" \
  "manifest=$MANIFEST_ABS" \
  "model=$MODEL_PATH" \
  "start_index=$START_INDEX" \
  "total_samples=$TOTAL_SAMPLES" \
  "num_partitions=$NUM_SHARDS" \
  "perturb_mode=$PERTURB_MODE" \
  "perturb_position=$PERTURB_POSITION" \
  "perturb_index=$PERTURB_INDEX" \
  "selection_seed=$SELECTION_SEED" \
  "perturb_seed=$PERTURB_SEED" \
  "iou_range=$IOU_MIN,$IOU_MAX" \
  "perturb_box_mode=$PERTURB_BOX_MODE" \
  "random_box_size=$RANDOM_BOX_MIN_SIZE,$RANDOM_BOX_MAX_SIZE" \
  "iou_threshold=$IOU_THRESHOLD" \
  "context_window_tokens=$CONTEXT_WINDOW_TOKENS" \
  "baseline_results=$BASELINE_RESULTS" \
  "max_new_tokens=$MAX_NEW_TOKENS" \
  "final_max_new_tokens=$FINAL_MAX_NEW_TOKENS" \
  "temperature=$TEMPERATURE")
if [[ ! -f "$PARTITION_PLAN_PATH" && -d "$RUN_ROOT/shards" ]] \
    && compgen -G "$RUN_ROOT/shards/shard_*/results.jsonl" >/dev/null; then
  echo "Existing shards have no partition plan: $RUN_ROOT/shards" >&2
  echo 'Use a new RUN_TAG/OUTPUT_ROOT so stale shards cannot be merged.' >&2
  exit 2
fi
if [[ -f "$PARTITION_PLAN_PATH" ]]; then
  EXISTING_PARTITION_PLAN=$(<"$PARTITION_PLAN_PATH")
  if [[ "$EXISTING_PARTITION_PLAN" != "$PARTITION_PLAN" ]]; then
    echo "Run configuration differs from the existing partition plan: $PARTITION_PLAN_PATH" >&2
    echo 'Use a new RUN_TAG/OUTPUT_ROOT; stale partitions are never merged automatically.' >&2
    exit 2
  fi
fi
printf '%s\n' "$PARTITION_PLAN" >"$PARTITION_PLAN_PATH"

if (( USE_SHARDS )); then
  mkdir -p "$RUN_ROOT/shards"
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

report_status() {
  local gpu label result_path completed_count
  if (( ${#ACTIVE_PID_BY_GPU[@]} == 0 )); then
    if (( next_task < NUM_SHARDS )); then
      echo "[$(date '+%F %T')] waiting for an allowed GPU with memory.used <= ${GPU_IDLE_MEMORY_MB} MB"
    fi
    return
  fi
  for gpu in "${!ACTIVE_PID_BY_GPU[@]}"; do
    label=${ACTIVE_LABEL_BY_GPU[$gpu]}
    if (( USE_SHARDS )); then
      result_path="$RUN_ROOT/shards/$label/results.jsonl"
    else
      result_path="$RUN_ROOT/results.jsonl"
    fi
    completed_count=0
    if [[ -f "$result_path" ]]; then
      completed_count=$(wc -l < "$result_path")
    fi
    echo "[$(date '+%F %T')] running gpu=$gpu task=$label completed_records=$completed_count"
  done
}

launch_task() {
  local gpu=$1
  local shard=$next_task
  local start=${TASK_STARTS[$shard]}
  local count=${TASK_COUNTS[$shard]}
  local label task_dir
  label="shard_$(printf '%03d' "$shard")"
  if (( USE_SHARDS )); then
    task_dir="$RUN_ROOT/shards/$label"
  else
    label=run
    task_dir="$RUN_ROOT"
  fi
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
  elif [[ "$EXPERIMENT" == oracle ]]; then
    extra_args+=(--context-window-tokens "$CONTEXT_WINDOW_TOKENS")
    eval_script=eval/Oracle_experiment/gqa/evaluate_online_oracle.py
  elif [[ "$EXPERIMENT" == selective_router ]]; then
    extra_args+=(
      --baseline-results "$BASELINE_RESULTS"
      --iou-threshold "$IOU_THRESHOLD"
      --context-window-tokens "$CONTEXT_WINDOW_TOKENS"
      --verifier-log "$task_dir/verifier_events.jsonl"
      --model-load-lock "$RUN_ROOT/artifacts/model_load.lock"
      --model-load-lock-timeout-seconds "$MODEL_LOAD_LOCK_TIMEOUT_SECONDS"
    )
    eval_script=eval/Oracle_experiment/gqa/evaluate_selective_oracle_router.py
  else
    extra_args+=(
      --model-load-lock "$RUN_ROOT/artifacts/model_load.lock"
      --model-load-lock-timeout-seconds "$MODEL_LOAD_LOCK_TIMEOUT_SECONDS"
    )
    eval_script=eval/Oracle_experiment/gqa/evaluate_testdev_baseline.py
  fi
  echo "[$(date '+%F %T')] launching gpu=$gpu shard=$shard start=$start count=$count"
  (
    cd "$PROJECT_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$VOCOT_PYTHON" -u "$eval_script" \
      --model-path "$MODEL_PATH" --manifest-path "$MANIFEST_PATH" --output "$output_path" \
      --run-id "$RUN_TAG" --run-split "$SPLIT_NAME" \
      --start-index "$start" --max-samples "$count" \
      --max-new-tokens "$MAX_NEW_TOKENS" --final-max-new-tokens "$FINAL_MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" "${extra_args[@]}"
  ) >"$log_path" 2>&1 &
  ACTIVE_PID_BY_GPU[$gpu]=$!
  ACTIVE_LABEL_BY_GPU[$gpu]="$label"
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
    report_status
    sleep "$POLL_SECONDS"
  fi
done

if (( FAILED_TASKS > 0 )); then
  if (( USE_SHARDS )); then
    echo "$FAILED_TASKS shard job(s) failed; inspect $RUN_ROOT/shards/*/run.log." >&2
  else
    echo "$FAILED_TASKS GQA job(s) failed; inspect $RUN_ROOT/run.log." >&2
  fi
  exit 1
fi

if (( ! USE_SHARDS )); then
  echo "[$(date '+%F %T')] GQA run finished: $RUN_ROOT"
  exit 0
fi

if [[ "$EXPERIMENT" == counterfactual ]]; then
  settings=$(printf '{"dataset":"gqa_val_manifest","sharded":true,"model_path":"%s","manifest_path":"%s","start_index":%s,"total_samples":%s,"num_partitions":%s,"perturb_mode":"%s","perturb_position":"%s","perturb_index":%s,"selection_seed":%s,"perturb_seed":%s,"iou_range":[%s,%s],"perturb_box_mode":"%s","random_box_size_range":[%s,%s],"max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s}' \
    "$MODEL_PATH" "$MANIFEST_ABS" "$START_INDEX" "$TOTAL_SAMPLES" "$NUM_SHARDS" \
    "$PERTURB_MODE" "$PERTURB_POSITION" "${PERTURB_INDEX:-null}" "$SELECTION_SEED" "$PERTURB_SEED" \
    "$IOU_MIN" "$IOU_MAX" "$PERTURB_BOX_MODE" "$RANDOM_BOX_MIN_SIZE" "$RANDOM_BOX_MAX_SIZE" \
    "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE")
elif [[ "$EXPERIMENT" == oracle ]]; then
  settings=$(printf '{"dataset":"gqa_val_manifest","sharded":true,"model_path":"%s","manifest_path":"%s","start_index":%s,"total_samples":%s,"num_partitions":%s,"oracle_mode":"online_explicit_target_oracle","oracle_box_coordinate_system":"normalized_xyxy_on_center_padded_square","context_window_tokens":%s,"max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s}' \
    "$MODEL_PATH" "$MANIFEST_ABS" "$START_INDEX" "$TOTAL_SAMPLES" "$NUM_SHARDS" \
    "$CONTEXT_WINDOW_TOKENS" "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE")
elif [[ "$EXPERIMENT" == selective_router ]]; then
  settings=$(printf '{"dataset":"gqa_val_manifest","sharded":true,"model_path":"%s","manifest_path":"%s","start_index":%s,"total_samples":%s,"num_partitions":%s,"mode":"online_selective_oracle_router_grounder","iou_threshold":%s,"baseline_results":"%s","unmatched_policy":"unverifiable_accept","oracle_box_coordinate_system":"normalized_xyxy_on_center_padded_square","context_window_tokens":%s,"max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s,"kv_cache":false,"model_load_serialized":true,"model_load_lock_timeout_seconds":%s}' \
    "$MODEL_PATH" "$MANIFEST_ABS" "$START_INDEX" "$TOTAL_SAMPLES" "$NUM_SHARDS" \
    "$IOU_THRESHOLD" "$BASELINE_RESULTS" "$CONTEXT_WINDOW_TOKENS" \
    "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE" \
    "$MODEL_LOAD_LOCK_TIMEOUT_SECONDS")
else
  settings=$(printf '{"dataset":"GQA","split":"testdev_balanced","sharded":true,"model_path":"%s","manifest_path":"%s","start_index":%s,"total_samples":%s,"num_partitions":%s,"mode":"untouched_vocot_baseline","answer_metric":"normalized_exact_match","gt_object_boxes_used":false,"max_new_tokens":%s,"final_max_new_tokens":%s,"temperature":%s,"model_load_serialized":true,"model_load_lock_timeout_seconds":%s}' \
    "$MODEL_PATH" "$MANIFEST_ABS" "$START_INDEX" "$TOTAL_SAMPLES" "$NUM_SHARDS" \
    "$MAX_NEW_TOKENS" "$FINAL_MAX_NEW_TOKENS" "$TEMPERATURE" \
    "$MODEL_LOAD_LOCK_TIMEOUT_SECONDS")
fi
cd "$PROJECT_ROOT"
aggregate_args=(
  --experiment "$EXPERIMENT"
  --shards-dir "$RUN_ROOT/shards"
  --expected-shards "$NUM_SHARDS"
  --output "$RUN_ROOT/results.jsonl"
  --settings "$settings"
  --run-id "$RUN_TAG"
  --run-split "$SPLIT_NAME"
)
if [[ "$EXPERIMENT" == selective_router ]]; then
  aggregate_args+=(--events-output "$RUN_ROOT/verifier_events.jsonl")
fi
"$VOCOT_PYTHON" eval/Oracle_experiment/gqa/aggregate_results.py "${aggregate_args[@]}"
echo "[$(date '+%F %T')] all GQA shards finished and merged: $RUN_ROOT"
