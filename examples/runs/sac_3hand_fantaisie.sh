#!/usr/bin/env bash
# Paper canonical SAC on 3-hand Fantaisie Impromptu (5M steps).
#
# Long virtuoso piece (episode = 800 steps, 2× For Elise). The 3rd hand
# has structural opportunity to engage during presto-arpeggio passages
# where |K_t| momentarily exceeds 2 hands' reach.
#
# UTD=1.0 paper-matched: gradient_steps=8 = n_envs=8.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-FantaisieImpromptu-ThreeHandPrototype-v0" \
    --experiment-name "sac_3hand_fantaisie" \
    --n-envs 8 \
    --gradient-steps 8 \
    --device cuda \
    "$@"
