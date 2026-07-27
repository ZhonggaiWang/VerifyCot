#!/usr/bin/env bash
# Multi-GPU GQA online-oracle wrapper.  See run_gqa_sharded_eval.sh for env vars.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_gqa_sharded_eval.sh" oracle
