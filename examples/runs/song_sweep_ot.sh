#!/usr/bin/env bash
# Song sweep: PPO + SAC (SB3, library defaults) on
#   - every 2-hand OT piece in omnipiano.envs._SONG_SWEEP_OT_PIECES (5M steps)
#   - every extra env in omnipiano.envs._SONG_SWEEP_EXTRA_ENVS (own budget,
#     currently the two 3-hand GymnopedieNo1 tasks at 7M)
# Goal: per-piece reward<->F1 correlation. seed=0, one run per (env, algo).
# Logs land in examples/logs/{algo}_sweep_{name}_seed0_N/.
#
# Usage:
#   examples/runs/song_sweep_ot.sh                # PPO then SAC, everything, sequential
#   examples/runs/song_sweep_ot.sh ppo            # only PPO
#   examples/runs/song_sweep_ot.sh sac Partita    # only SAC, envs whose id contains "Partita"
#   examples/runs/song_sweep_ot.sh "ppo sac" Gymnopedie
#   SEED=1 examples/runs/song_sweep_ot.sh         # other seed
#
# Runs are SEQUENTIAL on purpose: each PPO run spins up 16 SubprocVecEnv
# workers, each SAC run 24. Launching several at once spikes RAM at
# construction and can OOM the box. If you want concurrency, start a second
# copy of this script a couple of minutes after the first, with a disjoint
# filter.
set -euo pipefail
cd "$(dirname "$0")/../.."

ALGOS="${1:-ppo sac}"
FILTER="${2:-}"
SEED="${SEED:-0}"

# "env_id:total_steps" pairs
JOBS=$(python - <<'PY'
from omnipiano.envs import _SONG_SWEEP_OT_PIECES, _SONG_SWEEP_EXTRA_ENVS
for p, _, _ in _SONG_SWEEP_OT_PIECES:
    print(f"OmniPiano-{p}-FingeringOT-v0:5000000")
for env, steps in _SONG_SWEEP_EXTRA_ENVS:
    print(f"{env}:{steps}")
PY
)

for algo in $ALGOS; do
  for job in $JOBS; do
    env="${job%%:*}"; steps="${job##*:}"
    [[ -n "$FILTER" && "$env" != *"$FILTER"* ]] && continue
    # OmniPiano-Foo-FingeringOT-v0 -> foo ; OmniPiano-Foo-ThreeHand-StaticPartition-v0 -> foo_threehand_staticpartition
    short=$(echo "$env" | sed -e 's/^OmniPiano-//' -e 's/-v0$//' -e 's/-FingeringOT$//' | tr 'A-Z-' 'a-z_')
    name="${algo}_sweep_${short}_seed${SEED}"
    echo "=== [$(date '+%F %T')] $algo  $env  seed=$SEED  steps=$steps"
    python examples/run_sb3_baseline.py \
      --algo "$algo" --env "$env" --experiment-name "$name" \
      --seed "$SEED" --total-steps "$steps" \
      2>&1 | tee "examples/logs/${name}.log"
  done
done
