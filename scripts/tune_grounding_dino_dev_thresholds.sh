#!/usr/bin/env bash
# Tune Grounding DINO's top-1 box threshold on controlled GQA dev.
#
# Example:
#   GPU_ID=7 scripts/tune_grounding_dino_dev_thresholds.sh
#
# Re-run model inference instead of reusing the cache:
#   GPU_ID=7 FORCE_INFERENCE=1 scripts/tune_grounding_dino_dev_thresholds.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

GPU_ID=${GPU_ID:-7}
PYTHON_BIN=${PYTHON_BIN:-/home/zhonggai/miniconda3/envs/qwen25/bin/python}
BENCHMARK=${BENCHMARK:-output/verifier_benchmark/gqa_controlled/v1/benchmark.jsonl}
MODEL_PATH=${MODEL_PATH:-weights/grounding-dino-base}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
OUTPUT_DIR=${OUTPUT_DIR:-}
CACHE_DIR=${CACHE_DIR:-}
RUN_ID=${RUN_ID:-${RUN_TAG:-}}
BOX_THRESHOLDS=${BOX_THRESHOLDS:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
TEXT_THRESHOLD=${TEXT_THRESHOLD:-0.25}
SELECTION_METRIC=${SELECTION_METRIC:-macro_f1}
DINO_DTYPE=${DINO_DTYPE:-float32}
FORCE_INFERENCE=${FORCE_INFERENCE:-0}

args=(
  -u
  -m grounding_control.benchmarks.gqa_controlled.tune_grounding_dino_thresholds
  --benchmark "$BENCHMARK"
  --model-path "$MODEL_PATH"
  --output-root "$OUTPUT_ROOT"
  --box-thresholds "$BOX_THRESHOLDS"
  --text-threshold "$TEXT_THRESHOLD"
  --selection-metric "$SELECTION_METRIC"
  --dtype "$DINO_DTYPE"
  --device cuda:0
)
if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$CACHE_DIR" ]]; then
  args+=(--cache-dir "$CACHE_DIR")
fi
if [[ -n "$RUN_ID" ]]; then
  args+=(--run-id "$RUN_ID")
fi
if [[ "$FORCE_INFERENCE" == "1" ]]; then
  args+=(--force-inference)
fi

echo "Environment: qwen25"
echo "Physical GPU: $GPU_ID"
echo "Threshold grid: $BOX_THRESHOLDS"
echo "Fixed text threshold: $TEXT_THRESHOLD"
if [[ -n "$OUTPUT_DIR" ]]; then
  echo "Output (exact legacy directory): $OUTPUT_DIR"
else
  echo "Output: canonical run below $OUTPUT_ROOT/verifier_benchmark/runs/gqa_controlled_v1_dev/threshold_search"
fi
if [[ -n "$RUN_ID" ]]; then
  echo "Run id: $RUN_ID"
fi
if [[ -n "$CACHE_DIR" ]]; then
  echo "Inference cache (exact): $CACHE_DIR"
elif [[ -z "$OUTPUT_DIR" ]]; then
  echo "Inference cache: parameter-addressed below $OUTPUT_ROOT/verifier_benchmark/cache"
else
  echo "Inference cache: $OUTPUT_DIR/inference_cache"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" exec "$PYTHON_BIN" "${args[@]}"
