#!/usr/bin/env bash
# VStar end-to-end routing: oracle binary verifier + real Qwen7B Grounder.
#
# Full run:
#   GENERATOR_GPU=6 QWEN_GPU=7 RUN_ID=oracle_v_qwen7b_v1 \
#     scripts/run_vstar_oracle_verifier_qwen_grounder.sh
#
# One-sample smoke:
#   GENERATOR_GPU=6 QWEN_GPU=7 SAMPLE_ID=main:9 \
#     RUN_ID=smoke_main9 scripts/run_vstar_oracle_verifier_qwen_grounder.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

GENERATOR_GPU=${GENERATOR_GPU:-6}
QWEN_GPU=${QWEN_GPU:-7}
VOCOT_PYTHON=${VOCOT_PYTHON:-/home/zhonggai/miniconda3/envs/vocot/bin/python}
QWEN_PYTHON=${QWEN_PYTHON:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}

MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/data/zhonggai/models/Qwen2.5-VL-7B-Instruct}
BASELINE_RESULTS=${BASELINE_RESULTS:-output/vstar/online_oracle/full_238_padding_fix/results.jsonl}
IMAGE_DIR=${IMAGE_DIR:-/data/zhonggai/VStar}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
RUN_SPLIT=${RUN_SPLIT:-full_238}
RUN_ID=${RUN_ID:-${RUN_TAG:-}}
VERIFIER_LOG=${VERIFIER_LOG:-}

ORACLE_IOU_THRESHOLD=${ORACLE_IOU_THRESHOLD:-0.5}
REJECT_THRESHOLD=${REJECT_THRESHOLD:-0.25}
ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD:-0.75}
CONTEXT_WINDOW_TOKENS=${CONTEXT_WINDOW_TOKENS:-48}

QWEN_DTYPE=${QWEN_DTYPE:-bfloat16}
QWEN_MAX_NEW_TOKENS=${QWEN_MAX_NEW_TOKENS:-64}
QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-3136}
QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-12000000}
QWEN_ATTN_IMPLEMENTATION=${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}
QWEN_PROMPT_PROTOCOL=${QWEN_PROMPT_PROTOCOL:-compact_json_v1}
QWEN_BOUNDARY_TOLERANCE_PIXELS=${QWEN_BOUNDARY_TOLERANCE_PIXELS:-1.0}
WORKER_TIMEOUT=${WORKER_TIMEOUT:-600}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
LIKELIHOOD_REDUCTION=${LIKELIHOOD_REDUCTION:-mean}
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_ID=${SAMPLE_ID:-}
MISSING_EXPERT_POLICY=${MISSING_EXPERT_POLICY:-fail_open}
FAIL_FAST=${FAIL_FAST:-1}
NO_RESUME=${NO_RESUME:-0}
VERBOSE=${VERBOSE:-1}
DRY_RUN=${DRY_RUN:-0}

for gpu_name in GENERATOR_GPU QWEN_GPU; do
  gpu_value=${!gpu_name}
  if [[ ! "$gpu_value" =~ ^[0-9]+$ ]]; then
    echo "$gpu_name must be one physical GPU index, got: $gpu_value" >&2
    exit 2
  fi
done
if [[ "$GENERATOR_GPU" == "$QWEN_GPU" ]]; then
  echo "GENERATOR_GPU and QWEN_GPU must be different physical GPUs." >&2
  exit 2
fi
for flag_name in FAIL_FAST NO_RESUME VERBOSE DRY_RUN; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1, got: $flag_value" >&2
    exit 2
  fi
done
if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "START_INDEX must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer when set." >&2
  exit 2
fi
if [[ -n "$SAMPLE_ID" && ( "$START_INDEX" != 0 || -n "$MAX_SAMPLES" ) ]]; then
  echo "SAMPLE_ID cannot be combined with START_INDEX/MAX_SAMPLES." >&2
  exit 2
fi
case "$MISSING_EXPERT_POLICY" in
  fail_open|error) ;;
  *)
    echo "MISSING_EXPERT_POLICY must be fail_open or error." >&2
    exit 2
    ;;
esac
case "$LIKELIHOOD_REDUCTION" in
  mean|sum) ;;
  *)
    echo "LIKELIHOOD_REDUCTION must be mean or sum." >&2
    exit 2
    ;;
esac
case "$QWEN_PROMPT_PROTOCOL" in
  compact_json_v1|single_object_json_v2) ;;
  *)
    echo "Unsupported QWEN_PROMPT_PROTOCOL: $QWEN_PROMPT_PROTOCOL" >&2
    exit 2
    ;;
esac

for executable in "$VOCOT_PYTHON" "$QWEN_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Python executable not found: $executable" >&2
    exit 1
  fi
done
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "VoCoT model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$QWEN_MODEL_PATH" ]]; then
  echo "Qwen model not found: $QWEN_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_RESULTS" ]]; then
  echo "Padding-fixed VStar baseline not found: $BASELINE_RESULTS" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "VStar image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

args=(
  -u
  eval/grounding_control/vstar/evaluate_oracle_verifier_qwen_grounder.py
  --model-path "$MODEL_PATH"
  --baseline-results "$BASELINE_RESULTS"
  --image-dir "$IMAGE_DIR"
  --output-root "$OUTPUT_ROOT"
  --run-split "$RUN_SPLIT"
  --qwen-python "$QWEN_PYTHON"
  --qwen-model-path "$QWEN_MODEL_PATH"
  --qwen-gpu "$QWEN_GPU"
  --qwen-dtype "$QWEN_DTYPE"
  --qwen-max-new-tokens "$QWEN_MAX_NEW_TOKENS"
  --qwen-min-pixels "$QWEN_MIN_PIXELS"
  --qwen-max-pixels "$QWEN_MAX_PIXELS"
  --qwen-attn-implementation "$QWEN_ATTN_IMPLEMENTATION"
  --qwen-prompt-protocol "$QWEN_PROMPT_PROTOCOL"
  --qwen-boundary-tolerance-pixels "$QWEN_BOUNDARY_TOLERANCE_PIXELS"
  --worker-timeout "$WORKER_TIMEOUT"
  --oracle-iou-threshold "$ORACLE_IOU_THRESHOLD"
  --reject-threshold "$REJECT_THRESHOLD"
  --accept-threshold "$ACCEPT_THRESHOLD"
  --context-window-tokens "$CONTEXT_WINDOW_TOKENS"
  --missing-expert-policy "$MISSING_EXPERT_POLICY"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --likelihood-reduction "$LIKELIHOOD_REDUCTION"
  --start-index "$START_INDEX"
)
if [[ -n "$RUN_ID" ]]; then args+=(--run-id "$RUN_ID"); fi
if [[ -n "$VERIFIER_LOG" ]]; then args+=(--verifier-log "$VERIFIER_LOG"); fi
if [[ -n "$MAX_SAMPLES" ]]; then args+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "$SAMPLE_ID" ]]; then args+=(--sample-id "$SAMPLE_ID"); fi
if [[ "$FAIL_FAST" == 1 ]]; then args+=(--fail-fast); fi
if [[ "$NO_RESUME" == 1 ]]; then args+=(--no-resume); fi
if [[ "$VERBOSE" == 1 ]]; then args+=(--verbose); fi

echo "Experiment: oracle binary verifier + Qwen2.5-VL-7B Grounder"
echo "VoCoT generator: physical GPU $GENERATOR_GPU"
echo "Qwen Grounder: physical GPU $QWEN_GPU; max_pixels=$QWEN_MAX_PIXELS"
echo "Oracle verifier: reject when matched candidate IoU < $ORACLE_IOU_THRESHOLD"
echo "Unmatched reference: fail-open accept candidate"
echo "Missing Grounder policy: $MISSING_EXPERT_POLICY"
if [[ -n "$SAMPLE_ID" ]]; then
  echo "Population: sample_id=$SAMPLE_ID"
elif [[ -n "$MAX_SAMPLES" ]]; then
  echo "Population: start=$START_INDEX; max_samples=$MAX_SAMPLES"
else
  echo "Population: full padding-fixed VStar set from index $START_INDEX"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "$GENERATOR_GPU"
  printf '%q ' "$VOCOT_PYTHON" "${args[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GENERATOR_GPU" PYTHONUNBUFFERED=1 \
  exec "$VOCOT_PYTHON" "${args[@]}"
