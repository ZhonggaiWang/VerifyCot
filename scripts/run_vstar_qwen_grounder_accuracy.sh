#!/usr/bin/env bash
# Intrinsic VStar object-localization accuracy for Qwen2.5-VL Grounder.
#
# The parent evaluator is CPU-only.  It starts a persistent Qwen worker whose
# CUDA_VISIBLE_DEVICES is restricted to QWEN_GPU and sends only the clean image
# plus VStar's canonical target-object reference.
#
# Full run:
#   QWEN_GPU=7 RUN_ID=qwen7b_full_v1 \
#     scripts/run_vstar_qwen_grounder_accuracy.sh
#
# Three-target smoke:
#   QWEN_GPU=7 MAX_TARGETS=3 RUN_ID=qwen7b_smoke \
#     scripts/run_vstar_qwen_grounder_accuracy.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

QWEN_GPU=${QWEN_GPU:-7}
VOCOT_PYTHON=${VOCOT_PYTHON:-/home/zhonggai/miniconda3/envs/vocot/bin/python}
QWEN_PYTHON=${QWEN_PYTHON:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/data/zhonggai/models/Qwen2.5-VL-7B-Instruct}
ORACLE_BOXES_PATH=${ORACLE_BOXES_PATH:-output/vstar/annotations/oracle_boxes/full_238.jsonl}
IMAGE_DIR=${IMAGE_DIR:-/data/zhonggai/VStar}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}

RUN_SPLIT=${RUN_SPLIT:-full_238}
RUN_ID=${RUN_ID:-${RUN_TAG:-}}
METHOD_NAME=${METHOD_NAME:-qwen25_vl_7b}
SETTING=${SETTING:-}

QWEN_DTYPE=${QWEN_DTYPE:-bfloat16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MIN_PIXELS=${MIN_PIXELS:-3136}
# 12 MP retains high-resolution detail while bounding Qwen7B visual-encoder
# memory on a single 24 GiB card.  This is about 30x the retired 401408 cap.
# Use a Qwen-specific name so an unrelated MAX_PIXELS cannot affect the run.
QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-12000000}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
PROMPT_PROTOCOL=${PROMPT_PROTOCOL:-compact_json_v1}
BOUNDARY_TOLERANCE_PIXELS=${BOUNDARY_TOLERANCE_PIXELS:-1.0}
WORKER_TIMEOUT=${WORKER_TIMEOUT:-600}

START_TARGET_INDEX=${START_TARGET_INDEX:-0}
MAX_TARGETS=${MAX_TARGETS:-}
TARGET_ID=${TARGET_ID:-}
NO_RESUME=${NO_RESUME:-0}
FAIL_FAST=${FAIL_FAST:-1}
VERBOSE=${VERBOSE:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! "$QWEN_GPU" =~ ^[0-9]+$ ]]; then
  echo "QWEN_GPU must be one physical GPU index, got: $QWEN_GPU" >&2
  exit 2
fi
for flag_name in NO_RESUME FAIL_FAST VERBOSE DRY_RUN; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1, got: $flag_value" >&2
    exit 2
  fi
done
if [[ ! "$START_TARGET_INDEX" =~ ^[0-9]+$ ]]; then
  echo "START_TARGET_INDEX must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$MAX_TARGETS" && ! "$MAX_TARGETS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TARGETS must be a positive integer when set." >&2
  exit 2
fi
if [[ -n "$TARGET_ID" && ( "$START_TARGET_INDEX" != 0 || -n "$MAX_TARGETS" ) ]]; then
  echo "TARGET_ID cannot be combined with START_TARGET_INDEX/MAX_TARGETS." >&2
  exit 2
fi
case "$PROMPT_PROTOCOL" in
  compact_json_v1|single_object_json_v2) ;;
  *)
    echo "Unsupported PROMPT_PROTOCOL: $PROMPT_PROTOCOL" >&2
    exit 2
    ;;
esac

for executable in "$VOCOT_PYTHON" "$QWEN_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Python executable not found: $executable" >&2
    exit 1
  fi
done
if [[ ! -d "$QWEN_MODEL_PATH" ]]; then
  echo "Qwen model directory not found: $QWEN_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "VStar image directory not found: $IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -f "$ORACLE_BOXES_PATH" ]]; then
  echo "VStar oracle boxes not found: $ORACLE_BOXES_PATH" >&2
  exit 1
fi

args=(
  -u
  eval/grounding_control/vstar/evaluate_grounder_accuracy.py
  --backend qwen25_vl
  --oracle-boxes-path "$ORACLE_BOXES_PATH"
  --image-dir "$IMAGE_DIR"
  --worker-python "$QWEN_PYTHON"
  --model-path "$QWEN_MODEL_PATH"
  --gpu "$QWEN_GPU"
  --dtype "$QWEN_DTYPE"
  --worker-timeout "$WORKER_TIMEOUT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --min-pixels "$MIN_PIXELS"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --prompt-protocol "$PROMPT_PROTOCOL"
  --boundary-tolerance-pixels "$BOUNDARY_TOLERANCE_PIXELS"
  --run-split "$RUN_SPLIT"
  --method-name "$METHOD_NAME"
  --output-root "$OUTPUT_ROOT"
  --start-target-index "$START_TARGET_INDEX"
)
if [[ -n "$QWEN_MAX_PIXELS" ]]; then
  args+=(--max-pixels "$QWEN_MAX_PIXELS")
fi
if [[ -n "$RUN_ID" ]]; then args+=(--run-id "$RUN_ID"); fi
if [[ -n "$SETTING" ]]; then args+=(--setting "$SETTING"); fi
if [[ -n "$MAX_TARGETS" ]]; then args+=(--max-targets "$MAX_TARGETS"); fi
if [[ -n "$TARGET_ID" ]]; then args+=(--target-id "$TARGET_ID"); fi
if [[ "$NO_RESUME" == 1 ]]; then args+=(--no-resume); fi
if [[ "$FAIL_FAST" == 1 ]]; then args+=(--fail-fast); fi
if [[ "$VERBOSE" == 1 ]]; then args+=(--verbose); fi

echo "Benchmark: VStar intrinsic Grounder accuracy"
echo "Grounder: Qwen2.5-VL-7B; GPU=$QWEN_GPU; model=$QWEN_MODEL_PATH"
echo "Reference: canonical target_object; candidate/question/GT hidden"
echo "Prompt protocol: $PROMPT_PROTOCOL"
if [[ -n "$QWEN_MAX_PIXELS" ]]; then
  echo "Qwen pixel policy: explicit cap=$QWEN_MAX_PIXELS"
else
  echo "Qwen pixel policy: uncapped source resolution"
fi
if [[ -n "$TARGET_ID" ]]; then
  echo "Population: target_id=$TARGET_ID"
elif [[ -n "$MAX_TARGETS" ]]; then
  echo "Population: start=$START_TARGET_INDEX; max_targets=$MAX_TARGETS"
else
  echo "Population: all 292 annotated targets from index $START_TARGET_INDEX"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "$VOCOT_PYTHON" "${args[@]}"
  printf '\n'
  exit 0
fi

PYTHONUNBUFFERED=1 exec "$VOCOT_PYTHON" "${args[@]}"
