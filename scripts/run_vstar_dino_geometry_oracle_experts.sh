#!/usr/bin/env bash
# Run VoCoT and the Grounding DINO geometry verifier on separate physical GPUs.
#
# Typical full run:
#   GENERATOR_GPU=6 DINO_GPU=7 GEOMETRY_ACCEPT_IOU=0.6 \
#     RUN_ID=vstar_dino_iou06_v1 \
#     scripts/run_vstar_dino_geometry_oracle_experts.sh
#
# Inspect the resolved command without loading either model:
#   DRY_RUN=1 scripts/run_vstar_dino_geometry_oracle_experts.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

GENERATOR_GPU=${GENERATOR_GPU:-6}
DINO_GPU=${DINO_GPU:-7}
VOCOT_PYTHON=${VOCOT_PYTHON:-/home/zhonggai/miniconda3/envs/vocot/bin/python}
DINO_PYTHON=${DINO_PYTHON:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}

MODEL_PATH=${MODEL_PATH:-weights/Volcano-7b}
DINO_MODEL_PATH=${DINO_MODEL_PATH:-/data/zhonggai/models/grounding-dino-base}
BASELINE_RESULTS=${BASELINE_RESULTS:-output/vstar/online_oracle/full_238_padding_fix/results.jsonl}
IMAGE_DIR=${IMAGE_DIR:-/data/zhonggai/VStar}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
RUN_SPLIT=${RUN_SPLIT:-full_238}
RUN_ID=${RUN_ID:-${RUN_TAG:-}}
VERIFIER_LOG=${VERIFIER_LOG:-}

DINO_DTYPE=${DINO_DTYPE:-float32}
DINO_BOX_THRESHOLD=${DINO_BOX_THRESHOLD:-0.3}
DINO_TEXT_THRESHOLD=${DINO_TEXT_THRESHOLD:-0.25}
GEOMETRY_ACCEPT_IOU=${GEOMETRY_ACCEPT_IOU:-0.4}
GEOMETRY_CONTAINMENT=${GEOMETRY_CONTAINMENT:-0.7}
DINO_TOP_K_LOG=${DINO_TOP_K_LOG:-20}
VERIFIER_CONFIDENCE_THRESHOLD=${VERIFIER_CONFIDENCE_THRESHOLD:-0.0}
WORKER_TIMEOUT=${WORKER_TIMEOUT:-300}

CONTEXT_WINDOW_TOKENS=${CONTEXT_WINDOW_TOKENS:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
LIKELIHOOD_REDUCTION=${LIKELIHOOD_REDUCTION:-mean}
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}
SAMPLE_ID=${SAMPLE_ID:-}

# FAILURE_POLICY applies only to worker/runtime failures. Oracle experts keep
# their evaluator-defined fail-open policy when a reference has no unique GT
# match.
FAILURE_POLICY=${FAILURE_POLICY:-fail_fast}
MISSING_EXPERT_POLICY=${MISSING_EXPERT_POLICY:-fail_open}
NO_RESUME=${NO_RESUME:-0}
VERBOSE=${VERBOSE:-1}
DRY_RUN=${DRY_RUN:-0}

for gpu_name in GENERATOR_GPU DINO_GPU; do
  gpu_value=${!gpu_name}
  if [[ ! "$gpu_value" =~ ^[0-9]+$ ]]; then
    echo "$gpu_name must be one physical GPU index, got: $gpu_value" >&2
    exit 2
  fi
done
if [[ "$GENERATOR_GPU" == "$DINO_GPU" ]]; then
  echo "GENERATOR_GPU and DINO_GPU must be different physical GPUs." >&2
  exit 2
fi

for flag_name in NO_RESUME VERBOSE DRY_RUN; do
  flag_value=${!flag_name}
  if [[ "$flag_value" != 0 && "$flag_value" != 1 ]]; then
    echo "$flag_name must be 0 or 1, got: $flag_value" >&2
    exit 2
  fi
done
case "$FAILURE_POLICY" in
  fail_fast|fail-fast)
    FAILURE_POLICY=fail_fast
    ;;
  fail_open|fail-open)
    FAILURE_POLICY=fail_open
    ;;
  *)
    echo "FAILURE_POLICY must be fail_fast or fail_open, got: $FAILURE_POLICY" >&2
    exit 2
    ;;
esac
case "$MISSING_EXPERT_POLICY" in
  fail_open|error) ;;
  *)
    echo "MISSING_EXPERT_POLICY must be fail_open or error, got: $MISSING_EXPERT_POLICY" >&2
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
if [[ ! "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "START_INDEX must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer when set." >&2
  exit 2
fi
if [[ -n "$SAMPLE_ID" && ( "$START_INDEX" != 0 || -n "$MAX_SAMPLES" ) ]]; then
  echo "SAMPLE_ID cannot be combined with START_INDEX or MAX_SAMPLES." >&2
  exit 2
fi

if [[ ! -x "$VOCOT_PYTHON" ]]; then
  echo "vocot Python executable not found: $VOCOT_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$DINO_PYTHON" ]]; then
  echo "DINO Python executable not found: $DINO_PYTHON" >&2
  exit 1
fi
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "VoCoT model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$DINO_MODEL_PATH" ]]; then
  echo "Grounding DINO checkpoint directory not found: $DINO_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_RESULTS" ]]; then
  echo "Padding-fixed VStar baseline results not found: $BASELINE_RESULTS" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "VStar image directory not found: $IMAGE_DIR" >&2
  exit 1
fi

args=(
  -u
  eval/Oracle_experiment/vstar/evaluate_dino_geometry_oracle_experts.py
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
  --geometry-accept-iou "$GEOMETRY_ACCEPT_IOU"
  --geometry-containment "$GEOMETRY_CONTAINMENT"
  --dino-top-k-log "$DINO_TOP_K_LOG"
  --worker-timeout "$WORKER_TIMEOUT"
  --verifier-confidence-threshold "$VERIFIER_CONFIDENCE_THRESHOLD"
  --context-window-tokens "$CONTEXT_WINDOW_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --likelihood-reduction "$LIKELIHOOD_REDUCTION"
  --start-index "$START_INDEX"
  --missing-expert-policy "$MISSING_EXPERT_POLICY"
)
if [[ -n "$RUN_ID" ]]; then
  args+=(--run-id "$RUN_ID")
fi
if [[ -n "$VERIFIER_LOG" ]]; then
  args+=(--verifier-log "$VERIFIER_LOG")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$SAMPLE_ID" ]]; then
  args+=(--sample-id "$SAMPLE_ID")
fi
if [[ "$FAILURE_POLICY" == fail_open ]]; then
  args+=(--worker-fail-open)
else
  args+=(--fail-fast)
fi
if [[ "$NO_RESUME" == 1 ]]; then
  args+=(--no-resume)
fi
if [[ "$VERBOSE" == 1 ]]; then
  args+=(--verbose)
fi

echo "VoCoT generator: physical GPU $GENERATOR_GPU; Python $VOCOT_PYTHON"
echo "DINO verifier: physical GPU $DINO_GPU; Python $DINO_PYTHON"
echo "DINO checkpoint: $DINO_MODEL_PATH"
echo "Geometry: accept IoU=$GEOMETRY_ACCEPT_IOU; containment=$GEOMETRY_CONTAINMENT"
echo "DINO thresholds: box=$DINO_BOX_THRESHOLD; text=$DINO_TEXT_THRESHOLD"
echo "Worker failure policy: $FAILURE_POLICY"
echo "Missing oracle expert policy: $MISSING_EXPERT_POLICY"
if [[ -n "$SAMPLE_ID" ]]; then
  echo "Population: sample_id=$SAMPLE_ID"
elif [[ -n "$MAX_SAMPLES" ]]; then
  echo "Population: start=$START_INDEX; max_samples=$MAX_SAMPLES"
else
  echo "Population: full padding-fixed VStar set from index $START_INDEX"
fi
if [[ -n "$RUN_ID" ]]; then
  echo "Run id: $RUN_ID"
else
  echo "Run id: evaluator-generated timestamp"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "$GENERATOR_GPU"
  printf '%q ' "$VOCOT_PYTHON" "${args[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GENERATOR_GPU" PYTHONUNBUFFERED=1 \
  exec "$VOCOT_PYTHON" "${args[@]}"
