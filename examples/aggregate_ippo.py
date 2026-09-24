"""Aggregate N-seed IPPO runs into paper-grade mean±std + curves.

Usage:
    python examples/aggregate_ippo.py <run_dir> [<run_dir> ...]

Reads each run's eval_summary.json (terminal metrics) and
periodic_eval.jsonl (eval-over-time). Prints a mean±std table and
saves training/eval curve PNGs next to the first run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# The four metrics the benchmark treats as authoritative.
_METRICS = (
    "episode_task/musical_f1",
    "episode_task/musical_precision",
    "episode_task/musical_recall",
    "episode_task/sustain_f1",
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _terminal_table(run_dirs: list[Path]) -> None:
    """Print mean±std of terminal metrics + team return across seeds."""
    rows: dict[str, list[float]] = {m: [] for m in _METRICS}
    returns: list[float] = []
    for run in run_dirs:
        summary = json.load(open(run / "eval_summary.json"))["summary"]
        for m in _METRICS:
            rows[m].append(summary[f"{m}_mean"])
        returns.append(summary["team_return_mean"])

    n = len(run_dirs)
    print(f"\n=== Terminal metrics over {n} seed(s) (mean ± std) ===")
    for m in _METRICS:
        vals = np.array(rows[m])
        print(f"  {m:38s}: {vals.mean():.4f} ± {vals.std():.4f}")
    r = np.array(returns)
    print(f"  {'team_return':38s}: {r.mean():.2f} ± {r.std():.2f}")


def _plot_training(run_dirs: list[Path], out: Path) -> None:
    """Sampled agent-sum return vs env steps, one line per seed."""
    plt.figure(figsize=(8, 5))
    for run in run_dirs:
        rows = _load_jsonl(run / "progress.jsonl")
        xs = [r["env_steps"] for r in rows if r.get("rllib_agent_sum_return_mean") is not None]
        ys = [r["rllib_agent_sum_return_mean"] for r in rows if r.get("rllib_agent_sum_return_mean") is not None]
        if xs:
            plt.plot(xs, ys, alpha=0.8, label=_seed_label(run))
    plt.xlabel("environment steps")
    plt.ylabel("sampled agent-sum return (train)")
    plt.title("IPPO training curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")


def _plot_eval_with_band(run_dirs: list[Path], out: Path) -> None:
    """Eval F1 vs env steps: per-seed lines + across-seed mean±std band."""
    per_seed_xy: list[tuple[np.ndarray, np.ndarray]] = []
    plt.figure(figsize=(8, 5))
    for run in run_dirs:
        rows = _load_jsonl(run / "periodic_eval.jsonl")
        xs = np.array([r["actual_env_step"] for r in rows])
        f1 = np.array([r["summary"]["episode_task/musical_f1_mean"] for r in rows])
        if len(xs):
            plt.plot(xs, f1, alpha=0.35, marker=".", label=_seed_label(run))
            per_seed_xy.append((xs, f1))

    # Mean±std band on the common step grid (assumes shared eval cadence).
    if len(per_seed_xy) >= 2:
        ref_x = per_seed_xy[0][0]
        stack = np.vstack([np.interp(ref_x, x, y) for x, y in per_seed_xy])
        mean, std = stack.mean(0), stack.std(0)
        plt.plot(ref_x, mean, color="black", lw=2, label="mean")
        plt.fill_between(ref_x, mean - std, mean + std, color="black", alpha=0.15)

    plt.xlabel("environment steps")
    plt.ylabel("musical F1 (deterministic eval)")
    plt.ylim(0, 1)
    plt.title("IPPO evaluation F1 (truth metric)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")


def _seed_label(run: Path) -> str:
    """Extract 'seedN' from the run dir name for legends."""
    for part in run.name.split("_"):
        if part.startswith("seed"):
            return part
    return run.name[:20]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/aggregate_ippo.py <run_dir> [<run_dir> ...]")
        return 1
    run_dirs = [Path(p).expanduser().resolve() for p in sys.argv[1:]]
    _terminal_table(run_dirs)
    base = run_dirs[0]
    _plot_training(run_dirs, base / "curve_training_allseeds.png")
    _plot_eval_with_band(run_dirs, base / "curve_eval_f1_band.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())