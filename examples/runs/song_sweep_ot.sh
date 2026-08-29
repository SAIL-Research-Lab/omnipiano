#!/usr/bin/env bash
# Song sweep: PPO + SAC (SB3, library defaults) on every 2-hand OT piece
# registered in the "Song sweep" block of omnipiano/envs/__init__.py.
# Goal: per-piece reward<->F1 correlation. seed=0, 5M steps, one run per
# (piece, algo). Logs land in examples/logs/{algo}_sweep_{piece}_seed0_N/.
#
# Usage:
#   examples/runs/song_sweep_ot.sh                # PPO then SAC, all pieces, sequential
#   examples/runs/song_sweep_ot.sh ppo            # only PPO
#   examples/runs/song_sweep_ot.sh sac Partita    # only SAC, pieces matching "Partita"
#   SEED=1 examples/runs/song_sweep_ot.sh         # other seed
#
# Runs are SEQUENTIAL on purpose: each PPO run spins up 16 SubprocVecEnv
# workers, each SAC run 24. Launching several at once spikes RAM at
# construction and can OOM the box. If you want concurrency, start a second
# copy of this script a couple of minutes after the first, with a disjoint
# piece filter.
set -euo pipefail
cd "$(dirname "$0")/../.."

ALGOS="${1:-ppo sac}"
FILTER="${2:-}"
SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-5000000}"

PIECES=$(python - <<'PY'
from omnipiano.envs import _SONG_SWEEP_OT_PIECES
print(" ".join(p for p, _, _ in _SONG_SWEEP_OT_PIECES))
PY
)

for algo in $ALGOS; do
  for piece in $PIECES; do
    [[ -n "$FILTER" && "$piece" != *"$FILTER"* ]] && continue
    env="OmniPiano-${piece}-FingeringOT-v0"
    name="${algo}_sweep_$(echo "$piece" | tr 'A-Z' 'a-z')_seed${SEED}"
    echo "=== [$(date '+%F %T')] $algo  $env  seed=$SEED  steps=$TOTAL_STEPS"
    python examples/run_sb3_baseline.py \
      --algo "$algo" --env "$env" --experiment-name "$name" \
      --seed "$SEED" --total-steps "$TOTAL_STEPS" \
      2>&1 | tee "examples/logs/${name}.log"
  done
done
