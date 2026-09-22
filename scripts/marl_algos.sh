#!/usr/bin/env bash
# Launch MARL algorithm runs. Self-contained: no tmux, no screen, no helper
# script. Jobs are nohup'd into examples/logs/ with a PID file each, so an SSH
# drop cannot kill them and `status`/`stop` work from any new shell.
#
#   bash scripts/marl_algos.sh flags                  # what CLI flags exist
#   DRY=1 bash scripts/marl_algos.sh smoke happo      # print commands, run none
#   bash scripts/marl_algos.sh smoke happo
#   PIECES=WinterWind bash scripts/marl_algos.sh pilot happo mappo
#   PIECES="WinterWind PicturesGreatKiev" bash scripts/marl_algos.sh run happo
#   bash scripts/marl_algos.sh status
#   bash scripts/marl_algos.sh stop
#
# DELIBERATELY DOES NOT PASS A REWARD-PENALTY FLAG.
# The coefficient lives in configs/marl_train_config_default.json, which is what the completed
# IPPO/MAPPO 10M runs used. Passing it on the command line would let HAPPO drift
# from its own baselines, and then a HAPPO-vs-MAPPO difference would be a
# difference in the reward function rather than in the update rule -- the one
# confound that would invalidate the whole comparison. `provenance` checks it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE=${1:-help}; shift 2>/dev/null || true
ALGOS=("$@")

PY=${PY:-python}
MODULE=${MODULE:-omnipiano.multiagent.train}
CFG=${CFG:-omnipiano/multiagent/configs/marl_train_config_default.json}
LOGROOT=${LOGROOT:-examples/logs}
JOBDIR=${JOBDIR:-.marl_jobs}
PIECES=${PIECES:-WinterWind}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-9}
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
DRY=${DRY:-0}
ENV_TEMPLATE=${ENV_TEMPLATE:-OmniPiano-%s-FourHand-MA-Duet-Territorial-v0}
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Flag names, all overridable. If your --help disagrees, `flags` says so and
# tells you which variable to set -- rather than aborting on one bad guess the
# way the previous version did.
F_ALGO=${F_ALGO:---algo}
F_ENV=${F_ENV:---env-id}
F_SEED=${F_SEED:---seed}
F_STEPS=${F_STEPS:---total-steps}
F_WORKERS=${F_WORKERS:---num-workers}
F_CONFIG=${F_CONFIG:---config}
F_EXP=${F_EXP:---allow-experimental}

case "$MODE" in smoke) STEPS=${STEPS:-4000};;
                pilot) STEPS=${STEPS:-100000};;
                run)   STEPS=${STEPS:-10000000};;
                *)     STEPS=${STEPS:-0};; esac

HELPCACHE=$(mktemp); trap 'rm -f "$HELPCACHE"' EXIT
_help () { $PY -m "$MODULE" --help >"$HELPCACHE" 2>&1 || true; }

_needs_experimental () {  # keep in sync with AlgoSpec.status
  case "$1" in mat|masac|ppo-monolithic) return 0;; esac
  return 1
}

_check_flags () {
  _help
  local missing=()
  for pair in "$F_ALGO:F_ALGO" "$F_ENV:F_ENV" "$F_SEED:F_SEED" \
              "$F_STEPS:F_STEPS" "$F_WORKERS:F_WORKERS" "$F_EXP:F_EXP"; do
    local flag=${pair%%:*} var=${pair##*:}
    grep -q -- "$flag" "$HELPCACHE" || missing+=("  $flag   -> set $var=--real-flag")
  done
  if ((${#missing[@]})); then
    echo "[abort] these flags are not in --help:"
    printf '%s\n' "${missing[@]}"
    echo
    echo "Discover the real names with:"
    echo "  $PY -m $MODULE --help | grep -E 'algo|env-id|seed|step|runner|worker|experimental'"
    return 1
  fi
}

case "$MODE" in

flags)
  _help
  echo "== flags this script relies on =="
  grep -E -- '--(algo|env-id|seed|total|step|num-env-runner|num-worker|config|allow-experimental)' \
    "$HELPCACHE" || echo "(none matched -- paste --help output)"
  echo
  echo "== reward / penalty flags (NOT passed by this script, JSON owns them) =="
  grep -i -- 'penal\|collision' "$HELPCACHE" || echo "(none)"
  ;;

provenance)
  # The single check that decides whether a HAPPO run is comparable at all.
  echo "== reward section of $CFG =="
  jq '.reward // "NO reward SECTION"' "$CFG"
  echo
  echo "== what the completed runs actually used =="
  for f in "$LOGROOT"/*/run_config.json; do
    jq -r '[.algo,
            (.effective_config.inter_agent_collision_penalty_coef // "absent"),
            (.effective_config.learner_class // "?")] | @tsv' "$f" 2>/dev/null
  done | sort -u | column -t
  echo
  echo "The penalty column MUST be identical across algos, or the comparison"
  echo "measures the reward function rather than the update rule."
  ;;

smoke|pilot|run)
  ((${#ALGOS[@]})) || { echo "[abort] no algos given"; exit 1; }
  _check_flags || exit 1
  mkdir -p "$JOBDIR" "$LOGROOT"

  launched=0
  for piece in $PIECES; do
    for algo in "${ALGOS[@]}"; do
      env_id=$(printf "$ENV_TEMPLATE" "$piece")
      tag="${MODE}_${algo//[^a-zA-Z0-9]/_}_${piece}_seed${SEED}"
      log="$LOGROOT/${tag}.console.log"
      pidf="$JOBDIR/${tag}.pid"

      cmd=($PY -m "$MODULE" "$F_ALGO" "$algo" "$F_ENV" "$env_id"
           "$F_SEED" "$SEED" "$F_STEPS" "$STEPS" "$F_WORKERS" "$NUM_WORKERS")
      [[ -f "$CFG" ]] && grep -q -- "$F_CONFIG" "$HELPCACHE" && cmd+=("$F_CONFIG" "$CFG")
      _needs_experimental "$algo" && cmd+=("$F_EXP")

      if [[ "$DRY" == "1" ]]; then
        echo "[dry] ${cmd[*]}"; continue
      fi

      # Throttle: this box has ~36 usable cores and each job takes
      # NUM_WORKERS+1 processes, so oversubscribing makes every job slower
      # without finishing any sooner.
      while (( $(ls "$JOBDIR"/*.pid 2>/dev/null | while read -r p; do
                   kill -0 "$(cat "$p")" 2>/dev/null && echo x; done | wc -l)
               >= MAX_CONCURRENT )); do sleep 30; done

      echo "[launch] $tag  -> $log"
      nohup "${cmd[@]}" >"$log" 2>&1 &
      echo $! > "$pidf"
      launched=$((launched+1))
      sleep 5   # stagger runtime startup and environment construction
    done
  done
  echo "[ok] launched $launched job(s). Detach freely -- nohup survives SSH drop."
  echo "     bash scripts/marl_algos.sh status"
  ;;

status)
  shopt -s nullglob
  found=0
  for pidf in "$JOBDIR"/*.pid; do
    found=1
    tag=$(basename "$pidf" .pid); pid=$(cat "$pidf")
    if kill -0 "$pid" 2>/dev/null; then state="RUNNING pid=$pid"; else state="EXITED"; fi
    printf '%-52s %s\n' "$tag" "$state"
    log="$LOGROOT/${tag}.console.log"
    [[ -f "$log" ]] && tail -n 2 "$log" | sed 's/^/      | /'
    # Proof-of-life that the custom learner is really in the loop. Absent =>
    # you are looking at MAPPO numbers under a HAPPO label.
    for p in "$LOGROOT"/*/progress.jsonl; do
      [[ "$p" == *"${tag#*_}"* ]] || continue
      n=$(grep -c 'compound_factor\|num_agents_in_permutation' "$p" 2>/dev/null || echo 0)
      echo "      | happo metrics rows: $n"
    done
  done
  ((found)) || echo "no jobs in $JOBDIR"
  ;;

stop)
  shopt -s nullglob
  for pidf in "$JOBDIR"/*.pid; do
    pid=$(cat "$pidf")
    kill -0 "$pid" 2>/dev/null && { echo "[kill] $(basename "$pidf" .pid) pid=$pid"; kill "$pid"; }
    rm -f "$pidf"
  done
  ;;

*) sed -n '2,30p' "$0"; exit 1;;
esac
