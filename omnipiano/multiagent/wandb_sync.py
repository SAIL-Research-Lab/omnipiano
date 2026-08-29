"""Backfill completed OmniPiano multi-agent run directories into W&B.

Replays ``progress.jsonl`` / ``periodic_eval.jsonl`` at their recorded lifetime
env-step values, so a backfilled curve is indistinguishable from a live one.

Every backfilled run is tagged ``backfill`` so pre-fix data can never be
silently mixed with corrected runs.

Globs are resolved against the current directory AND against the repository
root, so this works from any cwd::

    python -m omnipiano.multiagent.wandb_sync --run-dir 'examples/logs/ippo_*'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno} is not valid JSON ({exc}); skipping")
                continue
            if isinstance(row, dict):
                yield row


def _infer_group(run_config: Mapping[str, Any], run_dir: Path) -> str:
    algo = str(run_config.get("effective_config", {}).get("algo") or "").lower()
    if not algo:
        algo = "mappo" if run_config.get("is_centralized_critic") else "ippo"
    env_id = str(run_config.get("env_id", "unknown"))
    token = "_".join(
        p.lower() for p in env_id.split("-")
        if p and p != "OmniPiano" and not p.startswith("v")
    )
    return (
        f"{algo}__{token}"
        if token != "unknown"
        else f"{algo}__{run_dir.name}"
    )


def sync_run_dir(
    run_dir: Path,
    *,
    entity: Optional[str],
    project: str,
    mode: str,
    upload_videos: bool,
    dry_run: bool,
) -> Optional[str]:
    run_dir = run_dir.expanduser().resolve()
    run_config = _read_json(run_dir / "run_config.json")
    eval_summary = _read_json(run_dir / "eval_summary.json")
    if not run_config:
        print(f"[skip] {run_dir}: no run_config.json")
        return None

    group = _infer_group(run_config, run_dir)
    seed = run_config.get("seed")
    steps = run_config.get("actual_total_env_steps") or run_config.get("total_env_steps") or 0
    tags = sorted({
        group.split("_")[0],
        f"seed{seed}",
        "backfill",
        *(["smoke-test"] if run_config.get("smoke_test") else []),
        *([f"{int(steps)//1_000_000}M"] if int(steps) >= 1_000_000 else []),
    })

    progress = list(_read_jsonl(run_dir / "progress.jsonl"))
    evals = list(_read_jsonl(run_dir / "periodic_eval.jsonl"))
    print(
        f"[sync] {run_dir.name}\n"
        f"        group={group} seed={seed} "
        f"progress_rows={len(progress)} eval_rows={len(evals)}"
    )
    if dry_run:
        return None

    from omnipiano.multiagent._wandb import WandbRun

    wb = WandbRun(
        mode=mode,
        entity=entity,
        project=project,
        name=run_dir.name,
        group=group,
        job_type="backfill",
        tags=tags,
        notes=f"Backfilled from {run_dir}",
        config=run_config,
        run_dir=run_dir,
    )
    try:
        # Interleave train / eval rows in env-step order: W&B requires the
        # logging step to be non-decreasing.
        timeline: List[tuple] = []
        for row in progress:
            step = row.get("env_steps")
            if step is not None:
                timeline.append((int(step), 0, row))
        for row in evals:
            step = row.get("actual_env_step", row.get("scheduled_env_step"))
            if step is not None:
                timeline.append((int(step), 1, row))
        timeline.sort(key=lambda item: (item[0], item[1]))

        for step, kind, row in timeline:
            if kind == 0:
                wb.log_train(step, row, result=None)
            else:
                wb.log_eval(step, row, scheduled_env_step=row.get("scheduled_env_step"))

        if eval_summary:
            wb.log_final(int(eval_summary.get("actual_total_env_steps", steps)), eval_summary)
            for name in ("run_config.json", "eval_summary.json",
                         "progress.jsonl", "periodic_eval.jsonl"):
                wb.log_artifact_dir(
                    run_dir / name,
                    name=f"{run_dir.name}__{name.replace('.', '_')}",
                )
        if upload_videos and (run_dir / "videos").is_dir():
            wb.log_artifact_dir(
                run_dir / "videos", name=f"{run_dir.name}__videos",
                artifact_type="eval-videos",
            )
        url = wb.url
    finally:
        wb.finish()
    print(f"[done] {run_dir.name} -> {url}")
    return url


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True,
                        help="Run directory or glob. Repeatable.")
    parser.add_argument("--entity", default="omnipiano")
    parser.add_argument("--project", default="marl")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--upload-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be synced and exit.")
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pattern in args.run_dir:
        expanded: List[str] = []
        raw = str(Path(pattern).expanduser())
        expanded.extend(glob.glob(raw))
        if not Path(raw).is_absolute():
            # Also interpret the pattern relative to the repo root, so the same
            # command works from ~/, from the repo root, and from its parent.
            from omnipiano.multiagent.paths import repo_root
            expanded.extend(glob.glob(str(repo_root() / raw)))
        matches = sorted({p for p in expanded if Path(p).is_dir()})
        if not matches:
            print(f"[warn] no directory matched {pattern!r}")
        paths.extend(Path(p) for p in matches)
    if not paths:
        print("[error] nothing to sync. Try: --run-dir 'examples/logs/*'",
              file=sys.stderr)
        return 1

    for path in paths:
        sync_run_dir(
            path, entity=args.entity, project=args.project, mode=args.mode,
            upload_videos=args.upload_videos, dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
