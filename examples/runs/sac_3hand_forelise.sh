#!/usr/bin/env bash
# SAC on 3-hand For Elise (5M steps).
#
# Companion to the Winter Wind run: For Elise is a genuinely 2-hand piece
# (annotated for 2 hands, no |K_t| > 5 simultaneity), so this run serves
# as the empirical control showing that under plain OT the 3rd hand
# stays vestigial when the music doesn't need it. Compare F1 against
# sac_3hand_winter_wind to ablate "wide-range repertoire helps 3rd-hand
# engagement".
#
# UTD=1.0 paper-matched: gradient_steps=24 = n_envs=24.
# Paired with sac_3hand_winter_wind.sh (same 24/24 config) so the
# "wide-range piece engages 3rd hand vs ForElise control" ablation
# uses identical wall-clock parallelism — F1 differences are then
# attributable to repertoire choice, not compute footprint.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python examples/run_sb3_sac_template.py \
    --env "OmniPiano-ForElise-ThreeHandPrototype-v0" \
    --experiment-name "sac_3hand_forelise" \
    --n-envs 24 \
    --gradient-steps 24 \
    --device cuda \
    "$@"
