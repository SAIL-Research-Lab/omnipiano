#!/usr/bin/env bash
# TQC on 4-hand Winter Wind, Level-3 (no partition), 5M steps.
#
# This is the TQC counterpart of sac_4hand_winter_wind.sh — same env,
# same morphology, same hparam discipline (n_envs=24, gradient_steps=24,
# UTD=1.0, device=cuda). Only the algorithm changes: SAC → TQC.
#
# Pair with tqc_4hand_winter_wind_static_partition.sh for the
# Level-3 vs Level-1 partition ablation under TQC.
#
# Why TQC: SAC's twin-critic min-of-two systematically underestimates Q
# on long-horizon dense-reward tasks like piano-playing. TQC's truncated
# quantile critics give a tunable bias-variance knob and consistently
# beat SAC on continuous control benchmarks. See run_sb3_tqc_template.py
# header for details.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_tqc_template.py \
    --env "OmniPiano-WinterWind-FourHandPrototype-v0" \
    --experiment-name "tqc_4hand_winter_wind" \
    --n-envs 24 \
    --gradient-steps 24 \
    --device cuda \
    "$@"
