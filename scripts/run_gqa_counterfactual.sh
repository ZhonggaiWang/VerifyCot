#!/usr/bin/env bash
# GQA counterfactual wrapper (direct or data-parallel). See the scheduler for env vars.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_gqa_sharded_eval.sh" counterfactual
