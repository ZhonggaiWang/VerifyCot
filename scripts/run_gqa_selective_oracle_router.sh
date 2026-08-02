#!/usr/bin/env bash
# GQA selective oracle verifier+grounder wrapper (direct or data-parallel).
# Configure GPU_IDS, IOU_THRESHOLD, NUM_SHARDS, and other shared controls via
# environment variables; see run_gqa_sharded_eval.sh.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_gqa_sharded_eval.sh" selective_router
