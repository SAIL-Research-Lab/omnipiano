#!/usr/bin/env bash
# TQC on 5-hand Winter Wind, Level-1 (static partition), 8M steps.
#
# First Level-1 5-hand experiment. Tests whether the partition mechanism
# generalizes from N=4 to N=5 — at 5-hand the OT outer-hand-idle problem
# is even worse (25 fingertips compete for ~5 keys per timestep, leaving
# 20 fingers ungranted), so partition is the only known way to get all 5
# hands to participate.
#
# Algorithm choice: TQC (NOT SAC). SAC + Level-1 partition + n_envs ≥ 24
# previously exhibited entropy collapse (ent_coef → 0.001 within ~100
# updates → policy frozen at do-nothing). TQC's truncated quantile critics
# avoid this by giving softer Q signals.
#
# Hyperparameters: n_envs=32 / gradient_steps=32 (UTD=1.0) — maximum
# throughput on Ryzen 9 9950X (16 physical / 32 logical cores). Same
# parallelism as the 10M Level-3 TQC run which ran cleanly without the
# SAC entropy-collapse issue (TQC's truncated quantile critics are robust
# to high update rate under partition).
#
# Step budget 8M because 5-hand has higher action dimension (111 vs 89
# for 4-hand) → expect slower per-step convergence.
#
# Hardware: RTX 5090 + Ryzen 9 9950X. Estimated wall-clock 8-12 hours.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_tqc_template.py \
    --env "OmniPiano-WinterWind-FiveHand-StaticPartition-v0" \
    --experiment-name "tqc_5hand_winter_wind_static_partition_8M" \
    --total-steps 8000000 \
    --n-envs 32 \
    --gradient-steps 32 \
    --device cuda \
    "$@"
