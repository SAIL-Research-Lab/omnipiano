cat > scripts/marl_algos.sh <<'EOF'
#!/usr/bin/env bash
# Algorithm-comparison stage on the OmniPiano 4-hand MA Territorial suite.
# Penalty for cross-agent duplicate presses is ON by default this round.
#
#   bash scripts/marl_algos.sh algos                    # what is launchable
#   bash scripts/marl_algos.sh smoke happo              # 5k steps, in-process
#   bash scripts/marl_algos.sh pilot happo              # 100k, production topology
#   bash scripts/marl_algos.sh run happo ppo-monolithic # 4 jobs x 10M (~7-9 h)
#   bash scripts/marl_algos.sh status | gate | stop
#
# Off-policy (facmac/masac) is refused until implemented; see AlgoSpec.blocking.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIECES=${PIECES:-"WinterWind PicturesGreatKiev"}
SEED=${SEED:-0}
STEPS=${STEPS:-10000000}
EVAL_FREQ=${EVAL_FREQ:-50000}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-1800}
# Locked to the topology of the completed IPPO/MAPPO runs so that a cross-algo
# difference cannot be blamed on the sampling layout. Costs idle cores when
# fewer than 4 jobs run; that is the price of a controlled comparison.
NW=${NUM_WORKERS:-9}
PENALTY=${PENALTY:-1}
WANDB_PROJECT=${WANDB_PROJECT:-multiagent}
WANDB_MODE=${WANDB_MODE:-online}

LOGS=examples/logs; PIDS=$LOGS/.pids; mkdir -p "$PIDS"
PROC='omnipiano.multiagent.train'
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); (( NGPU < 1 )) && NGPU=1
env_id () { echo "OmniPiano-$1-FourHand-MA-Duet-Territorial-v0"; }
tag_of  () { tr -dc 'A-Z' <<<"$1" | tr 'A-Z' 'a-z'; }

# Never guess a CLI flag name. Discover it, or stop.
penalty_flag () {
  [[ "$PENALTY" == "1" ]] || { echo ""; return 0; }
  local help f
  help=$(MUJOCO_GL=egl python -m "$PROC" --help 2>/dev/null)
  for f in --duplicate-press-penalty --dup-press-penalty --penalty-duplicate-press \
           --coordination-penalty --enable-penalty; do
    grep -q -- "$f" <<<"$help" && { echo "$f"; return 0; }
  done
  echo "[abort] PENALTY=1 but no penalty flag found in --help. Candidates checked:" >&2
  echo "        --duplicate-press-penalty --dup-press-penalty ..." >&2
  echo "        Find the real one:  python -m $PROC --help | grep -i penal" >&2
  echo "        Then set PENALTY_FLAG=--your-flag, or PENALTY=0 to disable." >&2
  return 1
}

launch () {
  local algo=$1 piece=$2 idx=$3 gpu ptag run
  gpu=$(( idx % NGPU )); ptag=$(tag_of "$piece")
  run=$LOGS/${algo//-/_}_$(tr 'A-Z' 'a-z' <<<"$piece")_seed${SEED}_$(date +%Y%m%d-%H%M%S)
  mkdir -p "$run"
  echo "[launch] gpu=$gpu $algo $piece workers=$NW penalty='${PFLAG:-off}' -> $run"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  RAY_TMPDIR=/tmp/rl_${gpu}_${ptag}_$$ WANDB_MODE=$WANDB_MODE \
  nohup python -m "$PROC" \
      --algo "$algo" --env-id "$(env_id "$piece")" --seed "$SEED" \
      --run-dir "$run" --total-steps "$STEPS" --eval-freq "$EVAL_FREQ" \
      --num-workers "$NW" --ray-num-cpus $((NW+2)) --num-gpus-per-learner 1.0 \
      --sample-timeout-s "$SAMPLE_TIMEOUT" --allow-experimental ${PFLAG:-} \
      --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
      >"$run/train.log" 2>&1 &
  echo $! > "$PIDS/${algo//-/_}_${ptag}_seed${SEED}.pid"
}

CMD=${1:-algos}; shift || true
ALGOS=${*:-${ALGOS:-happo}}

case $CMD in
algos)
  MUJOCO_GL=egl python -m "$PROC" --list-algos ;;

smoke)
  PFLAG=${PENALTY_FLAG:-$(penalty_flag)} || exit 1
  MUJOCO_GL=egl python -m "$PROC" --algo "$(awk '{print $1}' <<<"$ALGOS")" \
    --env-id "$(env_id "$(awk '{print $1}' <<<"$PIECES")")" --smoke-test \
    --num-workers 0 --allow-experimental ${PFLAG:-} \
    --wandb-mode disabled --ray-log-to-driver \
    --run-dir "$LOGS/smoke_$(date +%Y%m%d-%H%M%S)" ;;

pilot)
  STEPS=100000 EVAL_FREQ=50000 bash "$0" run $ALGOS ;;

run)
  pgrep -f "$PROC" >/dev/null && { echo "[abort] training already alive; 'stop' first."; exit 1; }
  PFLAG=${PENALTY_FLAG:-$(penalty_flag)} || exit 1
  export PFLAG
  # Refuse a not-implemented algo BEFORE burning a GPU-day on it.
  for a in $ALGOS; do
    MUJOCO_GL=egl python - "$a" <<'PY' || exit 1
import sys
from omnipiano.multiagent.algos import get_algo
spec = get_algo(sys.argv[1])
try:
    spec.assert_launchable(allow_experimental=True)
except ValueError as e:
    print(f"[abort] {e}"); sys.exit(1)
print(f"[ok] {spec.name}: status={spec.status} family={spec.family} "
      f"execution={spec.execution} learner={spec.learner_class}")
PY
  done
  MUJOCO_GL=egl python -m "$PROC" --list-envs 2>/dev/null > /tmp/_ma_envs.txt
  for p in $PIECES; do
    grep -qx "$(env_id "$p")" /tmp/_ma_envs.txt \
      || { echo "[abort] NOT registered: $(env_id "$p")"; exit 1; }
  done
  n=$(( $(wc -w <<<"$PIECES") * $(wc -w <<<"$ALGOS") ))
  echo "[plan] algos='$ALGOS' jobs=$n gpus=$NGPU workers/job=$NW cores=$(nproc) steps=$STEPS"
  i=0
  for a in $ALGOS; do for p in $PIECES; do launch "$a" "$p" "$i"; i=$((i+1)); sleep 8; done; done
  echo "[ok] W&B -> https://wandb.ai/omnipiano/$WANDB_PROJECT"
  echo "[ok] run inside tmux so an SSH drop cannot kill it" ;;

status|gate|stop)
  # Identical bookkeeping to marl10m.sh -- reuse it rather than duplicate it.
  STEPS=$STEPS SEED=$SEED bash scripts/marl10m.sh "$CMD" ;;

*) echo "usage: $0 {algos|smoke|pilot|run|status|gate|stop} [algo ...]"; exit 1 ;;
esac
EOF
chmod +x scripts/marl_algos.sh