#!/usr/bin/env python3
"""Fail-fast validation for the frozen WinterWind SCHO reproduction suite."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = (
    ROOT
    / "omnipiano/multiagent/configs/scho_winterwind_5x5x3"
)
ALGOS = ("ippo", "mappo", "happo", "a2po", "facmac")
TASKS = ("base", "observability", "coupling", "heterogeneous", "scalability")
SEEDS = (0, 1, 2)


def canonical_digest(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checksums(suite: Path) -> None:
    for line in (suite / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = suite / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {relative}")


def verify_dependencies(suite: Path) -> None:
    mismatches = []
    for raw in (suite / "requirements-reproduction.txt").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError("dependency mismatch:\n  " + "\n  ".join(mismatches))


def verify_dataset(suite: Path, *, required: bool) -> None:
    identity = load(suite / "dataset_identity.json")
    vendored = ROOT / "omnipiano/envs"
    sys.path.insert(0, str(vendored))
    try:
        from robopianist import music

        path = music._PIG_NAME_TO_FILE.get(identity["pig_piece_key"])
    except Exception as exc:  # pragma: no cover - installation-specific
        if required:
            raise RuntimeError(f"cannot inspect PIG dataset: {exc}") from exc
        print(f"DATASET: unavailable ({type(exc).__name__}: {exc})")
        return
    if path is None or not Path(path).is_file():
        if required:
            raise RuntimeError(
                f"PIG piece {identity['pig_piece_key']!r} is not installed"
            )
        print(f"DATASET: {identity['pig_piece_key']} not installed; runtime audit required")
        return
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = identity.get("midi_sha256")
    if expected is not None and digest != expected:
        raise RuntimeError(f"MIDI hash mismatch for {path}")
    print(f"DATASET: {path} sha256={digest}")


def verify_runtime_source(source_commit: str) -> None:
    resolved = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", source_commit],
        text=True,
        capture_output=True,
    )
    if resolved.returncode:
        raise RuntimeError(f"source commit is unavailable: {source_commit}")
    comparison = subprocess.run(
        [
            "git", "-C", str(ROOT), "diff", "--quiet", source_commit, "--",
            "omnipiano", ":(exclude)omnipiano/multiagent/configs",
        ]
    )
    if comparison.returncode:
        raise RuntimeError(
            "runtime source differs from the reviewed commit; checkout the "
            "recorded commit or audit the code change before reproducing"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--check-dependencies", action="store_true")
    parser.add_argument("--require-dataset", action="store_true")
    args = parser.parse_args()

    suite = args.suite.expanduser().resolve()
    manifest = load(suite / "manifest.json")
    provenance = load(suite / "provenance.json")
    verify_checksums(suite)
    if manifest["git_commit"] != provenance["reviewed_runtime_commit"]:
        raise RuntimeError("manifest/provenance runtime commits disagree")
    if tuple(manifest["selected_algorithms"]) != ALGOS:
        raise RuntimeError("manifest algorithm list is not the frozen baseline set")
    verify_runtime_source(provenance["reviewed_runtime_commit"])

    task_hashes = {}
    for task in TASKS:
        value = load(suite / "tasks" / f"{task}.json")
        task_hashes[task] = canonical_digest(value)
    if task_hashes != manifest["task_hashes"]:
        raise RuntimeError("task hashes do not match manifest")

    rows = []
    for path in sorted((suite / "runs").glob("*.json")):
        config = load(path)
        algo = config["experiment"]["algo"]
        seed = config["experiment"]["seed"]
        task = config["task"]["name"].removeprefix("scho-v2-winterwind-4h-")
        rows.append((algo, task, seed, path, config))

    expected = {(a, t, s) for a in ALGOS for t in TASKS for s in SEEDS}
    actual = {(a, t, s) for a, t, s, _, _ in rows}
    if len(rows) != 75 or actual != expected:
        raise RuntimeError(
            f"expected 75 unique algorithm/task/seed runs, got {len(rows)}"
        )

    manifest_rows = {item["file"]: item for item in manifest["runs"]}
    if len(manifest_rows) != 75:
        raise RuntimeError("manifest must contain exactly 75 unique run entries")
    for algo, task, seed, path, config in rows:
        relative = f"runs/{path.name}"
        item = manifest_rows.get(relative)
        if item is None or item["config_sha256"] != canonical_digest(config):
            raise RuntimeError(f"manifest mismatch: {relative}")
        if (item["algo"], item["setting"], item["seed"]) != (algo, task, seed):
            raise RuntimeError(f"identity mismatch: {relative}")
        if canonical_digest(config["task"]) != task_hashes[task]:
            raise RuntimeError(f"task snapshot mismatch: {relative}")
        if config["experiment"]["run_dir"] is not None:
            raise RuntimeError(f"frozen config is not portable: {relative}")
        if config["protocol"]["total_steps"] != 10_000_000:
            raise RuntimeError(f"wrong training budget: {relative}")
        if config["reward"]["inter_agent_collision_penalty_coef"] != 0.1:
            raise RuntimeError(f"wrong collision penalty: {relative}")
        video = config["video"]
        if not (
            video["enabled"] and video["record_final"]
            and video["wandb_upload"] and video["freq"] == 500_000
        ):
            raise RuntimeError(f"wrong video protocol: {relative}")

    if args.check_dependencies:
        verify_dependencies(suite)
    verify_dataset(suite, required=args.require_dataset)
    print(
        "REPRODUCTION_SUITE_OK: 25 algorithm/task cells, "
        "75 seeded runs, frozen runtime/config hashes verified"
    )


if __name__ == "__main__":
    main()
