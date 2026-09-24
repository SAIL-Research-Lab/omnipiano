"""Plan by default; --execute launches sequential, independent experiment cells."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omnipiano.safety.runtime import algorithm_config, atomic_json, code_hash, midi_hash, versions
from omnipiano.safety.suite import VERSION, SONGS, MAIN, HANDS, THRESHOLDS, EXTENSIONS, cells
from omnipiano.safety.algorithms import SUPPORTED_ALGORITHMS


def make_cell(group, index, *, device="cpu", smoke=False):
    item, algorithm, seed, steps = cells(group)[index]
    row = dict(version=VERSION, group=group, index=index, task=asdict(item), env_id=item.env_id,
               algorithm=algorithm, seed=seed, steps=2_000 if smoke else steps,
               device=device, smoke=smoke, steps_per_epoch=2_000 if smoke else 20_000,
               eval_episodes=1 if smoke else 10, code_hash=code_hash(), runtime=versions(),
               midi_hash=midi_hash(item))
    row["key"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    return row


def custom_cell(group, index, item, algorithm, seed, args):
    steps = args.steps if args.steps is not None else (2_000 if args.smoke_test else 2_000_000 if group == "extensions" else 5_000_000)
    cfg = algorithm_config(algorithm, item.budget, steps, seed, args.device,
                           smoke=args.smoke_test, max_ep_len=SONGS[item.song][1],
                           buffer_size=args.buffer_size)
    row = dict(version=VERSION, group=group, index=index, task=asdict(item), env_id=item.env_id,
               algorithm=algorithm, seed=seed, steps=steps, device=args.device,
               smoke=args.smoke_test, custom=True, algorithm_config=cfg,
               steps_per_epoch=cfg["algo_cfgs"]["steps_per_epoch"],
               save_model_freq=cfg["logger_cfgs"]["save_model_freq"],
               eval_episodes=args.eval_episodes or (1 if args.smoke_test else 10),
               evaluation_protocol="full-song-fixed-target-budget-v1",
               code_hash=code_hash(), runtime=versions(), midi_hash=midi_hash(item))
    row["key"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("main", "hands", "budget", "extensions"))
    parser.add_argument("--out", type=Path, default=Path("safety_runs"))
    parser.add_argument("--index", type=int, help="Zero-based cell; omit to plan/run the whole group")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="2K steps, one epoch, one replay episode")
    parser.add_argument("--list-algorithms", action="store_true")
    parser.add_argument("--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS)
    parser.add_argument("--task-index", type=int, nargs="+", help="Task indices, not legacy cell indices")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--buffer-size", type=int, help="Off-policy replay capacity; default 100000")
    args = parser.parse_args()
    if args.list_algorithms:
        print(f"{len(SUPPORTED_ALGORITHMS)} native algorithms:\n" + "\n".join(SUPPORTED_ALGORITHMS))
        return
    if args.group is None:
        parser.error("--group is required")
    if args.algorithms:
        if args.index is not None:
            parser.error("Use --task-index with --algorithms; --index belongs to the original matrix")
        if args.eval_episodes is not None and args.eval_episodes < 1:
            parser.error("--eval-episodes must be positive")
        tasks = {"main": MAIN, "hands": HANDS, "budget": THRESHOLDS, "extensions": EXTENSIONS}[args.group]
        indices = list(dict.fromkeys(args.task_index)) if args.task_index is not None else list(range(len(tasks)))
        seeds = list(dict.fromkeys(args.seeds)) if args.seeds is not None else ([0] if args.group == "extensions" else [1, 2, 3])
        if any(i < 0 or i >= len(tasks) for i in indices):
            parser.error("task-index outside the selected group")
        if args.smoke_test and (args.task_index is None or args.seeds is None):
            parser.error("Select --task-index and --seeds explicitly for a custom smoke test")
        rows = [custom_cell(args.group, i, tasks[i], a, s, args) for i in indices
                for a in dict.fromkeys(args.algorithms) for s in seeds]
        suffix = "custom_" + hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]
    else:
        if any(x is not None for x in (args.task_index, args.seeds, args.steps, args.eval_episodes, args.buffer_size)):
            parser.error("Custom options require --algorithms")
        if args.index is not None and not 0 <= args.index < len(cells(args.group)):
            parser.error("index outside the selected group")
        if args.smoke_test and args.index is None:
            parser.error("--smoke-test requires one explicit --index")
        indices = range(len(cells(args.group))) if args.index is None else [args.index]
        rows = [make_cell(args.group, i, device=args.device, smoke=args.smoke_test) for i in indices]
        suffix = "all" if args.index is None else str(args.index)
    manifest = args.out / f"{args.group}_{suffix}_{'smoke' if args.smoke_test else 'formal'}_manifest.json"
    atomic_json(manifest, rows)
    print(f"{len(rows)} cells; manifest: {manifest}; execute={args.execute}")
    for row in rows:
        print(row["index"], row["env_id"], row["algorithm"], row["seed"], row["steps"])
        if not args.execute:
            continue
        directory = args.out / args.group / row["key"]
        path = directory / "cell.json"
        if path.exists() and json.loads(path.read_text()) != row:
            # JSON converts tuples to lists; compare canonical serialized values.
            if json.dumps(json.loads(path.read_text()), sort_keys=True) != json.dumps(row, sort_keys=True):
                raise ValueError("Manifest conflict")
        else:
            atomic_json(path, row)
        subprocess.run([sys.executable, "-m", "omnipiano.safety.train", str(directory)], check=True)


if __name__ == "__main__":
    main()
