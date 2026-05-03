#!/usr/bin/env bash
# SAC on 4-hand Winter Wind (5M steps).
#
# Identical algorithm + hardware config to sac_3hand_winter_wind.sh
# (n_envs=24, gradient_steps=24, UTD=1.0, device=cuda) — only the
# morphology changes (3 hands -> 4 hands). This is the 3→4 step of
# the morphology ladder on identical repertoire (Chopin Etude Op.25
# No.11), letting us read off "does adding a 4th hand help on the
# same piece?" without confounding repertoire variation.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-WinterWind-FourHandPrototype-v0" \
    --experiment-name "sac_4hand_winter_wind" \
    --n-envs 24 \
    --gradient-steps 24 \
    --device cuda \
    "$@"
