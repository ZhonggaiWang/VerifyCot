#!/usr/bin/env bash
set -euo pipefail
# Prefer this wrapper when scheduling all three groups together: one scheduler
# owns every GPU, so independent group launchers cannot race for the same card.
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_vstar_counterfactual_sweep.sh" all
