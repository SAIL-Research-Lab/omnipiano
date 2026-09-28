#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SUITE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/examples/logs/scho_winterwind_5x5x3}"
AUDIT_DIR="$RUN_ROOT/_suite_audit"

mkdir -p "$AUDIT_DIR"
cd "$REPO_ROOT"

verify_args=("$SUITE_DIR" --require-dataset)
if [[ "${STRICT_DEPENDENCIES:-0}" == "1" ]]; then
  verify_args+=(--check-dependencies)
fi
"$PYTHON_BIN" omnipiano/multiagent/reproduction/verify_reproduction.py \
  "${verify_args[@]}"

"$PYTHON_BIN" omnipiano/multiagent/reproduction/audit_tasks.py \
  "$SUITE_DIR/runs" \
  --steps "${AUDIT_STEPS:-64}" \
  --output "$AUDIT_DIR/environment_audit.json"

if [[ "${1:-}" == "--verify-only" ]]; then
  echo "Verification complete; no training launched."
  exit 0
fi

unset WANDB_RUN_ID WANDB_NAME WANDB_RESUME

if [[ "$#" -eq 0 ]]; then
  set -- --gpus 0 --slots-per-gpu 1 --object-store-mb 4096 --stagger 60
fi

exec "$PYTHON_BIN" -u omnipiano/multiagent/reproduction/run_queue.py run \
  --config-dir "$SUITE_DIR/runs" \
  --run-root "$RUN_ROOT/runs" \
  "$@"
