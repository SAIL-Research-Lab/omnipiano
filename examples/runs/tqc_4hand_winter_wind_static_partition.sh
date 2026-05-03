#!/usr/bin/env bash
# TQC on 4-hand Winter Wind, Level-1 (static partition), 5M steps.
#
# Identical to tqc_4hand_winter_wind.sh EXCEPT:
#   --env: *-FourHand-StaticPartition-v0  (vs *-FourHandPrototype-v0)
#
# This is the partition-ON treatment for the Level-3 vs Level-1 ablation
# under TQC. Same algorithm, same hparams, same MIDI — only difference is
# whether each hand's forearm_tx is hard-clamped to its assigned key range
# (Level 1) or has full keyboard reach (Level 3).
#
# Expected outcome: under TQC at 5M steps, Level 1 should show all 4 hands
# engaging with their assigned regions (vs Level 3 outer-hand idle behavior
# observed under SAC).
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_tqc_template.py \
    --env "OmniPiano-WinterWind-FourHand-StaticPartition-v0" \
    --experiment-name "tqc_4hand_winter_wind_static_partition" \
    --n-envs 24 \
    --gradient-steps 24 \
    --device cuda \
    "$@"
