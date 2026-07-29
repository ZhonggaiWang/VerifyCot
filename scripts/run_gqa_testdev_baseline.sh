#!/usr/bin/env bash
# Multi-GPU untouched VoCoT baseline on official GQA Test-Dev balanced.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export MANIFEST_PATH=${MANIFEST_PATH:-$PROJECT_ROOT/output/gqa/annotations/testdev_balanced/manifest.jsonl}
exec "$SCRIPT_DIR/run_gqa_sharded_eval.sh" testdev_baseline
