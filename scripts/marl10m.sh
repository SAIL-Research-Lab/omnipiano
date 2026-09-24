#!/usr/bin/env bash
# 10M-step IPPO/MAPPO benchmark on the OmniPiano 4-hand MA Territorial suite.
# 1 seed, 2 pieces, 4 concurrent jobs:  IPPO -> GPU0 (both pieces),
#                                       MAPPO -> GPU1 (both pieces).
#
#   bash scripts/marl10m.sh envs    # registered MA env ids
#   bash scripts/marl10m.sh smoke   # 5k steps, in-process, no W&B     (~2 min)
#   bash scripts/marl10m.sh pilot   # 100k steps, PRODUCTION topology  (~8 min)
#   bash scripts/marl10m.sh run     # the real 10M x 4                 (~7-9 h)
#   bash scripts/marl10m.sh status  # steps/s + ETA + RAM + GPU
#   bash scripts/marl10m.sh gate    # last 5 eval rows per run
#   bash scripts/marl10m.sh stop    # graceful: SIGINT -> W&B flush -> SIGKILL
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIECES=${PIECES:-"WinterWind PicturesGreatKiev"}
ALGOS=${ALGOS:-"ippo mappo"}
SEED=${SEED:-0}
STEPS=${STEPS:-10000000}
EVAL_FREQ=${EVAL_FREQ:-50000}           # protocol cadence; measured ~9s per eval
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-1800}  # RLlib's 60s default DISCARDS slow runners
WANDB_PROJECT=${WANDB_PROJECT:-multiagent}
WANDB_MODE=${WANDB_MODE:-online}

NJOBS=$(( $(wc -w <<<"$PIECES") * $(wc -w <<<"$ALGOS") ))
CORES=$(nproc)
NW=${NUM_WORKERS:-$(( CORES / NJOBS - 2 ))}   # + 1 driver + 1 learner per job
if (( NW < 1 )); then NW=1; fi

LOGS=examples/logs; PIDS=$LOGS/.pids; mkdir -p "$PIDS"
PROC='omnipiano.multiagent.train'
env_id () { echo "OmniPiano-$1-FourHand-MA-Duet-Territorial-v0"; }
gpu_of () { if [[ $1 == ippo* ]]; then echo 0; else echo 1; fi; }
tag_of  () { tr -dc 'A-Z' <<<"$1" | tr 'A-Z' 'a-z'; }   # PicturesGreatKiev -> pgk

launch () {
  local algo=$1 piece=$2 gpu ptag run
  gpu=$(gpu_of "$algo"); ptag=$(tag_of "$piece")
  run=$LOGS/${algo}_$(tr 'A-Z' 'a-z' <<<"$piece")_seed${SEED}_$(date +%Y%m%d-%H%M%S)
  mkdir -p "$run"
  echo "[launch] gpu=$gpu $algo $piece workers=$NW -> $run"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  RAY_TMPDIR=/tmp/rl_${gpu}_${ptag}_$$ WANDB_MODE=$WANDB_MODE \
  nohup python -m "$PROC" \
      --algo "$algo" --env-id "$(env_id "$piece")" --seed "$SEED" \
      --run-dir "$run" --total-steps "$STEPS" --eval-freq "$EVAL_FREQ" \
      --num-workers "$NW" --ray-num-cpus $((NW+2)) --num-gpus-per-learner 1.0 \
      --sample-timeout-s "$SAMPLE_TIMEOUT" \
      --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
      >"$run/train.log" 2>&1 &
  echo $! > "$PIDS/${algo}_${ptag}_seed${SEED}.pid"
}

case ${1:-run} in
envs)
  MUJOCO_GL=egl python -m "$PROC" --list-envs 2>/dev/null | grep -- '-MA-' ;;

smoke)
  MUJOCO_GL=egl python -m "$PROC" --algo ippo \
    --env-id "$(env_id "$(awk '{print $1}' <<<"$PIECES")")" --smoke-test \
    --wandb-mode disabled --ray-log-to-driver \
    --run-dir "$LOGS/smoke_$(date +%Y%m%d-%H%M%S)" ;;

pilot)
  STEPS=100000 EVAL_FREQ=50000 bash "$0" run ;;

run)
  if pgrep -f "$PROC" >/dev/null; then
    echo "[abort] a training process is already alive; run 'stop' first."; exit 1
  fi
  echo "[plan] jobs=$NJOBS workers/job=$NW cores=$CORES steps=$STEPS eval_freq=$EVAL_FREQ"
  MUJOCO_GL=egl python -m "$PROC" --list-envs 2>/dev/null > /tmp/_ma_envs.txt
  for p in $PIECES; do
    grep -qx "$(env_id "$p")" /tmp/_ma_envs.txt \
      || { echo "[abort] NOT registered: $(env_id "$p")"; exit 1; }
  done
  for a in $ALGOS; do for p in $PIECES; do launch "$a" "$p"; sleep 8; done; done
  echo "[ok] W&B    -> https://wandb.ai/omnipiano/$WANDB_PROJECT"
  echo "[ok] local  -> tail -f $LOGS/*_seed${SEED}_*/train.log"
  echo "[ok] resume -> run inside tmux/screen so an SSH drop cannot kill it" ;;

status)
  free -g | awk 'NR<=2'
  for d in "$LOGS"/*_seed${SEED}_*/; do
    f="${d%/}/progress.jsonl"; [ -s "$f" ] || continue
    python - "$f" "$STEPS" <<'PY'
import json, sys
p, target = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(p) if l.strip()]
if not rows:
    raise SystemExit
r = rows[-1]
name = p.split("/")[-2][:54]
steps = int(r.get("env_steps", 0))
wall = max(float(r.get("wall_seconds", 0.0)), 1e-9)
if steps <= 0:
    print(f"{name:54s} steps=0  *** NO SAMPLES YET ***")
    raise SystemExit
sps = steps / wall
print(f"{name:54s} steps={steps:>10,} {sps:7.1f}/s wall={wall/3600:5.2f}h "
      f"eta={max((target-steps)/sps/3600.0, 0.0):6.2f}h "
      f"ret={r.get('rllib_agent_sum_return_mean')}")
PY
  done
  for f in "$LOGS"/*_seed${SEED}_*/failure.json; do
    [ -e "$f" ] && { echo "[FAILED] $f"; cat "$f"; }
  done
  for f in "$LOGS"/*_seed${SEED}_*/interrupted.json; do
    [ -e "$f" ] && echo "[STOPPED BY USER] $f"
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader ;;

gate)
  for d in "$LOGS"/*_seed${SEED}_*/; do
    f="${d%/}/periodic_eval.jsonl"; [ -s "$f" ] || continue
    echo "== ${d%/}"
    python - "$f" <<'PY'
import json, sys
nan = float("nan")
for r in [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-5:]:
    s = r["summary"]
    print(f'   step={r["actual_env_step"]:>9,} '
          f'f1={s["episode_task/musical_f1_mean"]:.4f} '
          f'team_return={s["team_return_mean"]:9.1f} '
          f'dup={s.get("episode_coordination/common_area_duplicate_press_rate_mean", nan):.3f} '
          f'coll={s.get("episode_coordination/inter_agent_collision_step_rate_mean", nan):.3f}')
PY
  done ;;

stop)
  # SIGINT/SIGTERM both reach train.py's KeyboardInterrupt path (Patch 7):
  # W&B is flushed, interrupted.json is written, NO failure.json.
  for f in "$PIDS"/*.pid; do
    [ -e "$f" ] || continue
    kill -INT "$(cat "$f")" 2>/dev/null || true
  done
  echo "[stop] SIGINT sent; waiting up to 90s for a graceful W&B flush..."
  for _ in $(seq 90); do
    pgrep -f "$PROC" >/dev/null || break
    sleep 1
  done
  pkill -9 -f "$PROC" 2>/dev/null || true
  rm -f "$PIDS"/*.pid
  rm -rf /tmp/rl_[0-9]_*_[0-9]*
  echo "[stop] done" ;;

*) echo "usage: $0 {envs|smoke|pilot|run|status|gate|stop}"; exit 1 ;;
esac