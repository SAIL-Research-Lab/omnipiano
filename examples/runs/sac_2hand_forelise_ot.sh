#!/usr/bin/env bash
# SAC on 2-hand For Elise OT (5M steps).
#
# Pins the SAC template's defaults explicitly so the wrapper is
# self-documenting. n_envs=4 follows the paper's mild parallelism win
# over n_envs=1 without changing UTD-relevant logic (default
# gradient_steps=1 → UTD=0.25; raise gradient_steps for paper UTD=1.0).
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-ForElise-FingeringOT-v0" \
    --experiment-name "sac_2hand_forelise_ot" \
    --n-envs 4 \
    "$@"
