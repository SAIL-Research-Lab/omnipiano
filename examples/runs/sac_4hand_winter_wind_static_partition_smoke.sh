#!/usr/bin/env bash
# SMOKE training for Level-1 (static-partition) 4-hand Winter Wind.
#
# Goal: verify the partition mechanism is wired correctly end-to-end —
# training runs without errors, eval video shows all 4 hands moving WITHIN
# their assigned key_range region. NOT meant to reach paper-quality F1.
#
# Differs from sac_4hand_winter_wind.sh only in:
#   - env id      : *-StaticPartition-v0 instead of *-FourHandPrototype-v0
#   - total steps : 30_000 (vs 5_000_000) — short enough for ~10 min on GPU
#                   while still past learning_starts=5000 + a useful number
#                   of gradient updates (UTD=1.0 → 30K updates).
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-WinterWind-FourHand-StaticPartition-v0" \
    --experiment-name "sac_4hand_winter_wind_static_partition_smoke" \
    --n-envs 24 \
    --gradient-steps 24 \
    --total-steps 30000 \
    --device cuda \
    "$@"
