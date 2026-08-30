#!/usr/bin/env bash
# OmniPiano multi-agent experiment driver. One command per stage.
#
#   bash scripts/marl_run.sh envs       # list registered MA env ids + algos
#   bash scripts/marl_run.sh check      # unit tests + import sanity (~2 min)
#   bash scripts/marl_run.sh smoke      # 5k steps, both algos, CPU (~10 min)
#   bash scripts/marl_run.sh equiv      # prove MAPPO(own critic) == IPPO
#   bash scripts/marl_run.sh pilot      # 100k steps on GPU; pre-run gate
#   bash scripts/marl_run.sh paper      # 2 pieces x 2 algos x 3 seeds x 5M
#   bash scripts/marl_run.sh status     # live progress of every launched run
#   bash scripts/marl_run.sh stop       # kill everything launched by `paper`
#   bash scripts/marl_run.sh backfill   # push the 3 historical seeds to W&B
#
# Overridable: PIECES SEEDS TOTAL_STEPS NUM_WORKERS EVAL_FREQ WANDB_MODE
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"                       # every relative path below is repo-anchored
LOGROOT="$REPO/examples/logs"
PIDDIR="$LOGROOT/.pids"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

TRAIN="python -m omnipiano.multiagent.train"

# --- experiment matrix (override from the shell) ---------------------------
PIECES="${PIECES:-PicturesGreatKiev WinterWind}"
SEEDS="${SEEDS:-0 1 2}"
TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
ALGOS="${ALGOS:-ippo mappo}"
WANDB_MODE="${WANDB_MODE:-online}"

env_id () { echo "OmniPiano-$1-FourHand-MA-Duet-Territorial-v0"; }

die () { echo "[error] $*" >&2; exit 1; }

require_envs () {
  local missing=()
  for piece in $PIECES; do
    python - "$(env_id "$piece")" <<'PY' || missing+=("$piece")
import sys
from omnipiano.multiagent import list_parallel_envs
sys.exit(0 if sys.argv[1] in list_parallel_envs() else 1)
PY
  done
  if ((${#missing[@]})); then
    echo "[error] these pieces have NO registered 4-hand MA env: ${missing[*]}" >&2
    echo "[error] registered multi-agent env ids:" >&2
    python -c "from omnipiano.multiagent import list_parallel_envs;print('\n'.join('  '+e for e in list_parallel_envs()))" >&2
    exit 1
  fi
}

# CPU budget: each job runs 1 driver + 1 learner + NUM_WORKERS env runners.
plan_cpus () {
  local njobs="$1"
  local total; total="$(nproc)"
  if [[ -z "${NUM_WORKERS:-}" ]]; then
    NUM_WORKERS=$(( total / njobs - 2 ))
    (( NUM_WORKERS > 8 )) && NUM_WORKERS=8
    (( NUM_WORKERS < 1 )) && NUM_WORKERS=1
  fi
  CPUS_PER_JOB=$(( NUM_WORKERS + 2 ))
  (( CPUS_PER_JOB * njobs > total )) && die \
    "$njobs jobs x $CPUS_PER_JOB CPUs > $total available. Set NUM_WORKERS lower."
  echo "[plan] jobs=$njobs num_workers=$NUM_WORKERS cpus_per_job=$CPUS_PER_JOB / $total"
}

launch () {  # $1=gpu $2=algo $3=piece $4=seed
  local gpu="$1" algo="$2" piece="$3" seed="$4"
  local tag="${algo}_$(echo "$piece" | tr '[:upper:]' '[:lower:]')_seed${seed}"
  local run_dir="$LOGROOT/${tag}_$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$PIDDIR"
  echo "[launch] gpu=$gpu $algo $piece seed=$seed -> $run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
  RAY_TMPDIR="/tmp/ray_${tag}_$$" \
  nohup $TRAIN \
      --algo "$algo" --env-id "$(env_id "$piece")" --seed "$seed" \
      --total-steps "$TOTAL_STEPS" --eval-freq "$EVAL_FREQ" --num-eval-eps 1 \
      --checkpoint-freq 500000 \
      --num-workers "$NUM_WORKERS" --ray-num-cpus "$CPUS_PER_JOB" \
      --num-learners 1 --num-gpus-per-learner 1 \
      --run-dir "$run_dir" \
      --wandb-mode "$WANDB_MODE" --wandb-entity omnipiano --wandb-project marl \
      --wandb-tags "paper,5M" --wandb-upload-artifacts \
      > "${run_dir}.stdout.log" 2>&1 &
  echo "$!" > "$PIDDIR/${tag}.pid"
  sleep 15   # stagger MuJoCo/EGL init
}

case "${1:-}" in

envs)
  echo "=== registered multi-agent env ids ==="
  python -c "from omnipiano.multiagent import list_parallel_envs;print('\n'.join(list_parallel_envs()))"
  echo; echo "=== registered algorithms ==="
  $TRAIN --list-algos
  ;;

check)
  python -c "import omnipiano; print('omnipiano OK:', omnipiano.make_parallel)"
  pytest -x -q \
    omnipiano/multiagent/ctde_module_test.py \
    omnipiano/multiagent/global_state_test.py \
    omnipiano/multiagent/reset_determinism_test.py \
    omnipiano/multiagent/multiagent_test.py
  bash "$0" envs
  ;;

smoke)
  for algo in $ALGOS; do
    rm -rf "/tmp/smoke_$algo"
    $TRAIN --algo "$algo" --smoke-test \
      --env-id "$(env_id "$(echo "$PIECES" | awk '{print $1}')")" \
      --num-gpus-per-learner 0 --wandb-mode disabled --run-dir "/tmp/smoke_$algo"
    python -m omnipiano.multiagent.evaluate \
      --checkpoint "/tmp/smoke_$algo" --no-video
  done
  python - <<'PY'
import glob, json
for algo in ("ippo", "mappo"):
    run = f"/tmp/smoke_{algo}"
    try:
        a = json.load(open(f"{run}/eval_summary.json"))["summary"]
    except FileNotFoundError:
        continue
    b = json.load(open(sorted(glob.glob(f"{run}/standalone_eval_*/eval_summary.json"))[-1]))["summary"]
    for k in ("team_return_mean", "episode_task/musical_f1_mean"):
        d = abs(a[k] - b[k])
        print(f"{algo:6s} {k:38s} {a[k]:.6f} vs {b[k]:.6f}  delta={d:.2e}")
        assert d < 1e-6, f"{algo}: checkpoint round-trip is NOT deterministic"
print("SMOKE PASSED: train -> checkpoint -> standalone eval is bit-identical")
PY
  ;;

equiv)
  # MAPPO with a decentralized critic reads exactly the same bytes as IPPO;
  # the extra global-state block is never touched by any network, so the two
  # runs must agree to floating-point precision.
  E="$(env_id "$(echo "$PIECES" | awk '{print $1}')")"
  for pair in "ippo:/tmp/equiv_ippo" "mappo-own-critic:/tmp/equiv_mappo_own"; do
    algo="${pair%%:*}"; dir="${pair##*:}"; rm -rf "$dir"
    $TRAIN --algo "$algo" --env-id "$E" --seed 0 \
      --total-steps 40000 --eval-freq 40000 --num-eval-eps 1 \
      --num-workers 2 --train-batch-size 2000 --checkpoint-freq 0 \
      --num-gpus-per-learner 0 --wandb-mode disabled --run-dir "$dir"
  done
  python - <<'PY'
import json
rows = lambda p: [json.loads(l) for l in open(f"{p}/progress.jsonl") if l.strip()]
a, b = rows("/tmp/equiv_ippo"), rows("/tmp/equiv_mappo_own")
assert len(a) == len(b), f"iteration count differs: {len(a)} vs {len(b)}"
worst = 0.0
for ra, rb in zip(a, b):
    assert ra["env_steps"] == rb["env_steps"]
    for k in ("rllib_agent_sum_return_mean", "episode_length_mean"):
        va, vb = ra.get(k), rb.get(k)
        if va is None or vb is None: continue
        worst = max(worst, abs(va - vb) / max(1.0, abs(va)))
print(f"worst relative deviation = {worst:.3e}")
assert worst < 1e-6, (
    "IPPO and MAPPO(own critic) diverged. Something observation-dimension-"
    "dependent is in the pipeline, or the actor slice is wrong.")
print("EQUIV PASSED: MAPPO reduces exactly to IPPO when its critic is decentralized")
PY
  ;;

pilot)
  plan_cpus 2
  require_envs
  P="$(echo "$PIECES" | awk '{print $1}')"
  TOTAL_STEPS=100000 EVAL_FREQ=20000 launch 0 ippo  "$P" 0
  TOTAL_STEPS=100000 EVAL_FREQ=20000 launch 1 mappo "$P" 0
  echo
  echo "Pilot launched. GATE: learner/*/vf_explained_var must rise above ~0.2."
  echo "  https://wandb.ai/omnipiano/marl"
  ;;

paper)
  require_envs
  njobs=0
  for p in $PIECES; do for a in $ALGOS; do for s in $SEEDS; do njobs=$((njobs+1)); done; done; done
  [[ "$ALGOS" == "ippo mappo" ]] || die \
    "paper scheduler currently requires ALGOS='ippo mappo'"
  # Run one IPPO and one MAPPO job at a time. Launching the full 12-run
  # matrix concurrently would oversubscribe both GPUs and invalidate timing.
  plan_cpus 2
  echo "[plan] pieces=[$PIECES] algos=[$ALGOS] seeds=[$SEEDS] steps=$TOTAL_STEPS"
  echo "[plan] $njobs total runs; at most two concurrent jobs (IPPO->GPU0, MAPPO->GPU1)"
  read -r -p "Run $njobs jobs? [y/N] " ok; [[ "$ok" == "y" ]] || exit 1
  for piece in $PIECES; do
    for seed in $SEEDS; do
      launch 0 ippo "$piece" "$seed"
      ippo_pid="$(cat "$PIDDIR/ippo_$(echo "$piece" | tr '[:upper:]' '[:lower:]')_seed${seed}.pid")"
      launch 1 mappo "$piece" "$seed"
      mappo_pid="$(cat "$PIDDIR/mappo_$(echo "$piece" | tr '[:upper:]' '[:lower:]')_seed${seed}.pid")"
      pair_status=0
      wait "$ippo_pid" || pair_status=1
      wait "$mappo_pid" || pair_status=1
      (( pair_status == 0 )) || die \
        "one or both jobs failed for piece=$piece seed=$seed"
    done
  done
  echo
  echo "All paper jobs completed."
  echo "  dashboard : https://wandb.ai/omnipiano/marl"
  echo "  progress  : bash scripts/marl_run.sh status"
  echo "  stop      : bash scripts/marl_run.sh stop"
  ;;

status)
  for d in "$LOGROOT"/{ippo,mappo}_*/; do
    [[ -f "$d/progress.jsonl" ]] || continue
    echo "== $(basename "$d")"
    tail -1 "$d/progress.jsonl" | python -c "
import json,sys
r=json.load(sys.stdin)
sps=r['env_steps']/max(r['wall_seconds'],1)
print(f\"   steps={r['env_steps']:>10,}  return={r.get('rllib_agent_sum_return_mean')}  \"
      f\"{sps:.0f} steps/s  eta={(${TOTAL_STEPS}-r['env_steps'])/max(sps,1)/3600:.1f}h\")"
    [[ -f "$d/periodic_eval.jsonl" ]] && tail -1 "$d/periodic_eval.jsonl" | python -c "
import json,sys
r=json.load(sys.stdin); s=r['summary']
print(f\"   eval@{r['actual_env_step']:,}  F1={s['episode_task/musical_f1_mean']:.4f}  \"
      f\"team_return={s['team_return_mean']:.1f}\")" 2>/dev/null || true
    [[ -f "$d/failure.json" ]] && { echo "   !! FAILED:"; cat "$d/failure.json"; }
  done
  command -v nvidia-smi >/dev/null && nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used --format=csv
  ;;

stop)
  for f in "$PIDDIR"/*.pid; do
    [[ -f "$f" ]] || continue
    pid="$(cat "$f")"
    kill "$pid" 2>/dev/null && echo "killed $(basename "$f" .pid) (pid $pid)"
    rm -f "$f"
  done
  ;;

backfill)
  python -m omnipiano.multiagent.wandb_sync \
    --run-dir 'examples/logs/ippo_rllib_*_seed*' \
    --entity omnipiano --project marl "${@:2}"
  ;;

*)
  sed -n '2,20p' "$0"; exit 1 ;;
esac
