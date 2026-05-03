#!/usr/bin/env bash
# SAC on 4-hand Winter Wind, Level-1 (static partition), 8M steps,
# maximum-throughput config.
#
# This is the partition-ON treatment for the Level-3 vs Level-1 ablation.
# Pairs with the existing 10M Level-3 TQC run (tqc_4hand_winter_wind_10M_1)
# to test whether Level-1 partition breaks through the F1≈0.42 ceiling
# observed under both algos at Level-3.
#
# Algorithm choice rationale: SAC at this budget (vs TQC) — empirically
# at the same training step SAC reaches ~0.10 higher F1 on this task
# (see eval CSV comparison). 8M SAC > 10M TQC in F1 wall-clock budget.
#
# Hardware: RTX 5090 + Ryzen 9 9950X (16 cores / 32 threads).
# Same n_envs=32 / gradient_steps=32 / UTD=1.0 config as the TQC 10M run
# for apples-to-apples comparison of partition vs no-partition.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-WinterWind-FourHand-StaticPartition-v0" \
    --experiment-name "sac_4hand_winter_wind_static_partition_8M" \
    --total-steps 8000000 \
    --n-envs 32 \
    --gradient-steps 32 \
    --device cuda \
    "$@"
