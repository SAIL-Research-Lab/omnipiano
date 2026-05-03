#!/usr/bin/env bash
# TQC on 4-hand Winter Wind, Level-3 (no partition), 10M steps,
# maximum-throughput config.
#
# Differs from tqc_4hand_winter_wind.sh in:
#   --total-steps 10_000_000         (vs 5M)        — longer training
#   --n-envs 32                      (vs 24)        — saturate AMD 9950X SMT
#   --gradient-steps 32              (vs 24)        — keep UTD=1.0
#
# Hardware: RTX 5090 (32 GiB) + Ryzen 9 9950X (16 cores / 32 threads).
# n_envs=32 maps 1:1 onto logical cores; SubprocVecEnv workers contend
# slightly with the SAC training process and eval env on the same SMT
# threads, but in practice this still gains ~30% throughput vs n_envs=24
# on this CPU.
#
# Why TQC at this scale:
#   - 10M env-step budget → 10M gradient updates @ UTD=1.0
#   - SAC's twin-critic min-of-two underestimates Q on long-horizon
#     dense-reward tasks like piano. TQC's truncated quantile critics
#     give stable Q targets and handle longer training without divergence.
#   - L-R-L-R 4-hand morphology (default_four_hand_specs) at full keyboard
#     reach — Level-3 baseline against which Level-1 partition will be
#     compared.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_tqc_template.py \
    --env "OmniPiano-WinterWind-FourHandPrototype-v0" \
    --experiment-name "tqc_4hand_winter_wind_10M" \
    --total-steps 10000000 \
    --n-envs 32 \
    --gradient-steps 32 \
    --device cuda \
    "$@"
