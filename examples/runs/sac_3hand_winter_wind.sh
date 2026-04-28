#!/usr/bin/env bash
# SAC on 3-hand Chopin Étude Op.25 No.11 "Winter Wind" (5M steps).
#
# Hypothesis target: short (314 steps, ~15.8s) but extreme pitch range
# (69 semitones, MIDI 32-101 ≈ 5.75 octaves) and dense bursts (~16.5
# notes/sec). If the 3rd hand engages here while it stays vestigial on
# For Elise, that's evidence the OT signal is sufficient given a piece
# that genuinely needs >2 hands. If F1 lands close to For Elise's
# baseline → plain OT is insufficient and we need unbalanced OT or
# zone-restricted OT to force engagement.
#
# UTD=1.0 paper-matched: gradient_steps=24 = n_envs=24.
# Bumped from 8/8 to 24/24 for ~3x wall-clock speedup at unchanged
# UTD (algorithmic behavior preserved). Matches PPO's n_envs=24
# CPU footprint so cross-algo wall-clock comparisons aren't muddied
# by parallelism mismatches.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-WinterWind-ThreeHandPrototype-v0" \
    --experiment-name "sac_3hand_winter_wind" \
    --n-envs 24 \
    --gradient-steps 24 \
    --device cuda \
    "$@"
