#!/usr/bin/env bash
# PPO on 2-hand For Elise with annotation-based fingering reward (5M steps).
#
# Pins the PPO template defaults. PPO + dense shaping on piano is known
# to plateau around F1 ~0.25 (paper Fig 8); this is the published
# baseline rather than a research target — we ship it so users can
# reproduce the paper's PPO numbers as a ceiling on what plain on-policy
# methods can achieve before reaching for SAC/DroQ.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_template.py \
    --env "OmniPiano-ForElise-FingeringAnn-v0" \
    --experiment-name "ppo_2hand_forelise_ann" \
    --n-envs 24 \
    "$@"
