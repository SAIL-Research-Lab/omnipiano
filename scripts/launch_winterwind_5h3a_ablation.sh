#!/usr/bin/env bash
# 2x2 factorial ablation on ONE GPU: {ippo,mappo} x {collision 0.1, 0.0}.
# Config knobs come from configs/marl_train_config_default.json untouched;
# we override ONLY algo, the ablated penalty, and an explicit run-dir (anti-collision).
set -euo pipefail

ENV_ID="OmniPiano-WinterWind-FiveHand-MA-Trio-Territorial-v0"
SEED=0
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p marl_runs logs

launch () {  # $1=algo  $2=penalty  $3=tag
  local dir="marl_runs/${3}_seed${SEED}_${STAMP}"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
    nohup python -m omnipiano.multiagent.train \
      --algo "$1" \
      --env-id "$ENV_ID" \
      --seed "$SEED" \
      --inter-agent-collision-penalty-coef "$2" \
      --run-dir "$dir" \
      > "logs/${3}_${STAMP}.out" 2>&1 &
  echo "launched [$3]  pid=$!  run_dir=$dir  log=logs/${3}_${STAMP}.out"
  sleep 2   # stagger starts so timestamps/registry init never race
}

launch ippo  0.1 ippo_coll01
launch ippo  0.0 ippo_coll00
launch mappo 0.1 mappo_coll01
launch mappo 0.0 mappo_coll00

echo "---- all 4 launched. jobs: ----"
jobs -l
