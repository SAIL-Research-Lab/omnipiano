#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

ENVS = {
    "ww": "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0",
    "pgk": "OmniPiano-PicturesGreatKiev-FourHand-MA-Duet-Territorial-v0",
}


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def make_configs(args):
    base = json.loads((ROOT / args.base).read_text(encoding="utf-8"))
    if base.get("schema_version") != 1 or "task" in base:
        raise ValueError("Use the supplied schema-v1 env_id baseline template.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous sweep.

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    steps = 20_000 if args.preflight else 10_000_000
    phase = "preflight20k" if args.preflight else "10M"

    index = 0
    for seed in (1, 2):
        for piece, env_id in ENVS.items():
            for algo in ("ippo", "mappo"):
                cfg = copy.deepcopy(base)
                name = (
                    f"{algo}_{piece}_4h2a_c0p1_seed{seed}_{phase}_{stamp}"
                )

                cfg["experiment"].update(
                    algo=algo,
                    env_id=env_id,
                    seed=seed,
                    run_dir=str(ROOT / "examples/logs" / stamp / name),
                )
                cfg["protocol"]["total_steps"] = steps
                cfg["reward"]["inter_agent_collision_penalty_coef"] = 0.1

                cfg["compute"].update(
                    num_workers=args.workers,
                    ray_num_cpus=args.workers + 2,
                    num_learners=1,
                    num_gpus_per_learner=1.0,
                    smoke_test=False,
                )

                # Preflight uses the REAL GPU/worker configuration.
                # Do not enable smoke_test: its overrides would switch to CPU.
                if args.preflight:
                    cfg["protocol"]["eval_freq"] = 10_000
                    cfg["compute"]["checkpoint_freq"] = 10_000
                    cfg["compute"]["log_every_iters"] = 1

                cfg["video"].update(
                    enabled=False,
                    record_final=False,
                    wandb_upload=False,
                )

                tags = [
                    algo, piece, "4hands", "2agents", "collision-0.1",
                    f"seed{seed}", phase, f"workers{args.workers}",
                ]
                cfg["wandb"].update(
                    name=name,
                    group=(
                        f"{algo}_{piece}_4h2a_c0p1_{phase}_w{args.workers}"
                    ),
                    tags=",".join(tags),
                    notes=(
                        f"env={env_id}; seed={seed}; collision_coef=0.1; "
                        f"target_train_env_steps={steps}; "
                        f"workers={args.workers}; isolated-Ray GPU sharing."
                    ),
                )
                if args.preflight:
                    cfg["wandb"]["mode"] = "offline"

                save(out / f"{index:02d}_{name}.json", cfg)
                print(name)
                index += 1

    print(f"\nGenerated {index} configs in {out}")


def run_queue(args):
    directory = Path(args.config_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if not 1 <= args.slots_per_gpu <= 4:
        raise ValueError("Use 1..4 slots per GPU; 4 is the requested limit.")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Duplicate GPU IDs.")

    state = directory.with_name(directory.name + ".state")
    state.mkdir(parents=True, exist_ok=True)

    lock = (ROOT / ".marl_jobs.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another marl_jobs supervisor is already running.")

    # An interrupted supervisor may have left live jobs behind.
    # Refuse to guess whether they should be launched again.
    for started in state.glob("*.started.json"):
        name = started.name[:-len(".started.json")]
        if not (state / f"{name}.exit.json").exists():
            raise RuntimeError(
                f"Unresolved previous job: {name}. "
                "Inspect its PID/log before restarting the supervisor."
            )

    slots = [
        (gpu, slot)
        for slot in range(args.slots_per_gpu)
        for gpu in args.gpus
    ]
    active = {}
    next_launch = 0.0
    fast_failures = 0
    had_failure = False

    pidfile = state / "supervisor.pid"
    pidfile.write_text(str(os.getpid()) + "\n")

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def finish(slot, interrupted=False):
        nonlocal fast_failures, had_failure
        proc, info = active.pop(slot)
        elapsed = time.time() - info["started_at"]
        rc = proc.returncode
        save(
            state / f"{info['name']}.exit.json",
            {
                **info,
                "returncode": rc,
                "elapsed_seconds": elapsed,
                "interrupted": interrupted,
                "training_budget_verified": False,
            },
        )
        print(
            f"EXIT gpu={slot[0]} slot={slot[1]} rc={rc} "
            f"{info['name']}", flush=True
        )
        if rc != 0 or interrupted:
            had_failure = True
        if rc != 0 and elapsed < 180:
            fast_failures += 1

    try:
        print(f"Supervisor PID={os.getpid()}, state={state}", flush=True)

        while True:
            for slot, (proc, info) in list(active.items()):
                if proc.poll() is not None:
                    finish(slot)

            if fast_failures >= 2:
                (state / "DRAIN").touch()
                print(
                    "Two fast failures: stop launching new jobs; inspect logs.",
                    flush=True,
                )
                fast_failures = 0

            draining = (state / "DRAIN").exists()
            pending = [
                p for p in sorted(directory.glob("*.json"))
                if not (state / f"{p.stem}.started.json").exists()
            ]
            free_slots = [s for s in slots if s not in active]

            if (
                not draining and pending and free_slots
                and time.time() >= next_launch
            ):
                cfg_path = pending[0]
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for key in ("experiment", "protocol", "compute"):
                    if key not in cfg:
                        raise ValueError(f"{cfg_path}: missing {key}")

                # Optional sidecar for a DIFFERENT frozen code worktree.
                # It contains one absolute directory path, not training config.
                cwd_file = cfg_path.with_suffix(".cwd")
                cwd = (
                    Path(cwd_file.read_text().strip()).resolve()
                    if cwd_file.exists() else ROOT
                )
                if not (cwd / "omnipiano/multiagent/train.py").is_file():
                    raise ValueError(f"Invalid code directory: {cwd}")

                slot = free_slots[0]
                name = cfg_path.stem
                snapshot = state / f"{name}.input.json"
                shutil.copyfile(cfg_path, snapshot)

                ray_root = Path(os.environ.get("RAY_TMP_ROOT", tempfile.gettempdir())) / "omnipiano_ray"
                ray_root.mkdir(parents=True, exist_ok=True)
                ray_tmp = tempfile.mkdtemp(prefix="j-", dir=ray_root)
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=slot[0],
                    RAY_ADDRESS="local",
                    RAY_TMPDIR=ray_tmp,
                    MUJOCO_GL="egl",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1",
                    PYTHONUNBUFFERED="1",
                    # Default object-store cap, not a total RAM limit.
                    # Explicit object_store_memory in ray.init takes precedence.
                    RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(
                        args.object_store_mb * 1024 * 1024
                    ),
                )
                env["PYTHONPATH"] = (
                    str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
                )

                info = {
                    "name": name,
                    "gpu": slot[0],
                    "slot": slot[1],
                    "cwd": str(cwd),
                    "config": str(snapshot),
                    "config_sha256": hashlib.sha256(
                        snapshot.read_bytes()
                    ).hexdigest(),
                    "ray_tmp": ray_tmp,
                    "started_at": time.time(),
                }
                save(state / f"{name}.started.json", info)

                with (state / f"{name}.log").open("wb") as log:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u", "-m",
                            "omnipiano.multiagent.train", str(snapshot),
                        ],
                        cwd=cwd,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

                info["pid"] = proc.pid
                active[slot] = (proc, info)
                save(state / f"{name}.started.json", info)

                print(
                    f"START gpu={slot[0]} slot={slot[1]} "
                    f"pid={proc.pid} {name}", flush=True
                )
                next_launch = time.time() + args.stagger

            if not active:
                if draining or (not pending and not args.watch):
                    break

            time.sleep(1)

    finally:
        # Only signal process groups started by THIS supervisor.
        # Normal runner cleanup should also call algo.stop()/ray.shutdown().
        for proc, info in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

        deadline = time.time() + 60
        while (
            any(p.poll() is None for p, _ in active.values())
            and time.time() < deadline
        ):
            time.sleep(1)

        for slot, (proc, info) in list(active.items()):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
            finish(slot, interrupted=True)

        pidfile.unlink(missing_ok=True)
        lock.close()

    return 1 if had_failure else 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make")
    make.add_argument(
        "--base",
        default="omnipiano/multiagent/configs/marl_train_config_default.json",
    )
    make.add_argument("--out", required=True)
    make.add_argument("--workers", type=int, required=True)
    make.add_argument("--preflight", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--config-dir", required=True)
    run.add_argument("--gpus", nargs="+", default=["0", "1"])
    run.add_argument("--slots-per-gpu", type=int, default=4)
    run.add_argument("--stagger", type=float, default=15)
    run.add_argument("--object-store-mb", type=int, default=2048)
    run.add_argument("--watch", action="store_true")

    args = parser.parse_args()
    if args.command == "make":
        make_configs(args)
        return 0
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())