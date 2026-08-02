#!/usr/bin/env bash
# Untouched VoCoT baseline on GQA Test-Dev (direct or data-parallel).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_gqa_sharded_eval.sh" testdev_baseline
