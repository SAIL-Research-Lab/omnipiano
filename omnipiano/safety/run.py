"""Plan by default; --execute launches sequential, independent experiment cells."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omnipiano.safety.runtime import atomic_json, code_hash, midi_hash, versions
from omnipiano.safety.suite import VERSION, cells


def make_cell(group, index, *, device="cpu", smoke=False):
    item, algorithm, seed, steps = cells(group)[index]
    row = dict(version=VERSION, group=group, index=index, task=asdict(item), env_id=item.env_id,
               algorithm=algorithm, seed=seed, steps=2_000 if smoke else steps,
               device=device, smoke=smoke, steps_per_epoch=2_000 if smoke else 20_000,
               eval_episodes=1 if smoke else 10, code_hash=code_hash(), runtime=versions(),
               midi_hash=midi_hash(item))
    row["key"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("main", "hands", "budget", "extensions"), required=True)
    parser.add_argument("--out", type=Path, default=Path("safety_runs"))
    parser.add_argument("--index", type=int, help="Zero-based cell; omit to plan/run the whole group")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="2K steps, one epoch, one replay episode")
    args = parser.parse_args()
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
