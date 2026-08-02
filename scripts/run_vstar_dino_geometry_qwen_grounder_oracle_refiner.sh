#!/usr/bin/env bash
# DINO geometry verifier + Qwen2.5-VL Grounder + oracle BoxRefiner.
#
# Example:
#   GENERATOR_GPU=6 DINO_GPU=7 QWEN_GPU=5 \
#   GEOMETRY_ACCEPT_IOU=0.6 RUN_ID=vstar_qwen_grounder_iou06_v1 \
#     scripts/run_vstar_dino_geometry_qwen_grounder_oracle_refiner.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

GENERATOR_GPU=${GENERATOR_GPU:-6}
DINO_GPU=${DINO_GPU:-7}
QWEN_GPU=${QWEN_GPU:-5}
VOCOT_PYTHON=${VOCOT_PYTHON:-/home/zhonggai/miniconda3/envs/vocot/bin/python}
DINO_PYTHON=${DINO_PYTHON:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}
QWEN_PYTHON=${QWEN_PYTHON:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}

MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
DINO_MODEL_PATH=${DINO_MODEL_PATH:-/data/zhonggai/models/grounding-dino-base}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/data/zhonggai/models/Qwen2.5-VL-7B-Instruct}
BASELINE_RESULTS=${BASELINE_RESULTS:-output/vstar/online_oracle/full_238_padding_fix/results.jsonl}
IMAGE_DIR=${IMAGE_DIR:-/data/zhonggai/VStar}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
RUN_SPLIT=${RUN_SPLIT:-full_238}
RUN_ID=${RUN_ID:-${RUN_TAG:-}}
VERIFIER_LOG=${VERIFIER_LOG:-}

DINO_DTYPE=${DINO_DTYPE:-float32}
DINO_BOX_THRESHOLD=${DINO_BOX_THRESHOLD:-0.3}
DINO_TEXT_THRESHOLD=${DINO_TEXT_THRESHOLD:-0.25}
DINO_TOP_K_LOG=${DINO_TOP_K_LOG:-20}
GEOMETRY_ACCEPT_IOU=${GEOMETRY_ACCEPT_IOU:-0.4}
GEOMETRY_CONTAINMENT=${GEOMETRY_CONTAINMENT:-0.7}
DINO_WORKER_TIMEOUT=${DINO_WORKER_TIMEOUT:-300}

QWEN_DTYPE=${QWEN_DTYPE:-bfloat16}
QWEN_MAX_NEW_TOKENS=${QWEN_MAX_NEW_TOKENS:-64}
QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-3136}
QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-401408}
QWEN_ATTN_IMPLEMENTATION=${QWEN_ATTN_IMPLEMENTATION:-sdpa}
QWEN_PROMPT_PROTOCOL=${QWEN_PROMPT_PROTOCOL:-compact_json_v1}
QWEN_BOUNDARY_TOLERANCE_PIXELS=${QWEN_BOUNDARY_TOLERANCE_PIXELS:-1.0}
QWEN_WORKER_TIMEOUT=${QWEN_WORKER_TIMEOUT:-300}

VERIFIER_CONFIDENCE_THRESHOLD=${VERIFIER_CONFIDENCE_THRESHOLD:-0.0}
CONTEXT_WINDOW_TOKENS=${CONTEXT_WINDOW_TOKENS:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
LIKELIHOOD_REDUCTION=${LIKELIHOOD_REDUCTION:-mean}
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_ID=${SAMPLE_ID:-}

# DINO_FAILURE_POLICY controls verifier transport failures.  Qwen model or
# transport failures follow MISSING_EXPERT_POLICY in
# PrecommitGroundingController.
DINO_FAILURE_POLICY=${DINO_FAILURE_POLICY:-fail_fast}
MISSING_EXPERT_POLICY=${MISSING_EXPERT_POLICY:-fail_open}
FAIL_FAST=${FAIL_FAST:-0}
NO_RESUME=${NO_RESUME:-0}
VERBOSE=${VERBOSE:-1}
DRY_RUN=${DRY_RUN:-0}

for gpu_name in GENERATOR_GPU DINO_GPU QWEN_GPU; do
  gpu_value=${!gpu_name}
  if [[ ! "$gpu_value" =~ ^[0-9]+$ ]]; then
    echo "$gpu_name must be one physical GPU index, got: $gpu_value" >&2
    exit 2
  fi
done
if [[ "$GENERATOR_GPU" == "$DINO_GPU" || "$GENERATOR_GPU" == "$QWEN_GPU" ]]; then
  echo "GENERATOR_GPU must differ from both expert-worker GPUs." >&2
  exit 2
fi

for flag_name in FAIL_FAST NO_RESUME VERBOSE DRY_RUN; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1, got: $flag_value" >&2
    exit 2
  fi
done
case "$DINO_FAILURE_POLICY" in
  fail_fast|fail-fast) DINO_FAILURE_POLICY=fail_fast ;;
  fail_open|fail-open) DINO_FAILURE_POLICY=fail_open ;;
  *)
    echo "DINO_FAILURE_POLICY must be fail_fast or fail_open." >&2
    exit 2
    ;;
esac
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

for executable in "$VOCOT_PYTHON" "$DINO_PYTHON" "$QWEN_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Python executable not found: $executable" >&2
    exit 1
  fi
done
for directory in "$MODEL_PATH" "$DINO_MODEL_PATH" "$QWEN_MODEL_PATH" "$IMAGE_DIR"; do
  if [[ ! -d "$directory" ]]; then
    echo "Required directory not found: $directory" >&2
    exit 1
  fi
done
if [[ ! -f "$BASELINE_RESULTS" ]]; then
  echo "Padding-fixed VStar baseline not found: $BASELINE_RESULTS" >&2
  exit 1
fi

args=(
  -u
  eval/Oracle_experiment/vstar/evaluate_dino_geometry_qwen_grounder_oracle_refiner.py
  --model-path "$MODEL_PATH"
  --baseline-results "$BASELINE_RESULTS"
  --image-dir "$IMAGE_DIR"
  --output-root "$OUTPUT_ROOT"
  --run-split "$RUN_SPLIT"
  --dino-python "$DINO_PYTHON"
  --dino-model-path "$DINO_MODEL_PATH"
  --dino-gpu "$DINO_GPU"
  --dino-dtype "$DINO_DTYPE"
  --dino-box-threshold "$DINO_BOX_THRESHOLD"
  --dino-text-threshold "$DINO_TEXT_THRESHOLD"
  --dino-top-k-log "$DINO_TOP_K_LOG"
  --geometry-accept-iou "$GEOMETRY_ACCEPT_IOU"
  --geometry-containment "$GEOMETRY_CONTAINMENT"
  --dino-worker-timeout "$DINO_WORKER_TIMEOUT"
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
  --qwen-worker-timeout "$QWEN_WORKER_TIMEOUT"
  --verifier-confidence-threshold "$VERIFIER_CONFIDENCE_THRESHOLD"
  --context-window-tokens "$CONTEXT_WINDOW_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --likelihood-reduction "$LIKELIHOOD_REDUCTION"
  --start-index "$START_INDEX"
  --missing-expert-policy "$MISSING_EXPERT_POLICY"
)
if [[ -n "$RUN_ID" ]]; then args+=(--run-id "$RUN_ID"); fi
if [[ -n "$VERIFIER_LOG" ]]; then args+=(--verifier-log "$VERIFIER_LOG"); fi
if [[ -n "$MAX_SAMPLES" ]]; then args+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "$SAMPLE_ID" ]]; then args+=(--sample-id "$SAMPLE_ID"); fi
if [[ "$DINO_FAILURE_POLICY" == fail_open ]]; then
  args+=(--dino-worker-fail-open)
fi
if [[ "$FAIL_FAST" == 1 ]]; then args+=(--fail-fast); fi
if [[ "$NO_RESUME" == 1 ]]; then args+=(--no-resume); fi
if [[ "$VERBOSE" == 1 ]]; then args+=(--verbose); fi

echo "VoCoT generator: GPU $GENERATOR_GPU"
echo "DINO verifier: GPU $DINO_GPU; model=$DINO_MODEL_PATH"
echo "Qwen Grounder: GPU $QWEN_GPU; model=$QWEN_MODEL_PATH"
if [[ "$DINO_GPU" == "$QWEN_GPU" ]]; then
  echo "DINO and Qwen share GPU $DINO_GPU; run a small smoke test first."
fi
echo "Geometry: accept IoU=$GEOMETRY_ACCEPT_IOU; containment=$GEOMETRY_CONTAINMENT"
echo "Qwen protocol: $QWEN_PROMPT_PROTOCOL; candidate box hidden"
echo "Qwen failure policy: $MISSING_EXPERT_POLICY"
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
