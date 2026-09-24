"""Plot IPPO training + evaluation curves from a run directory.

Usage:
    python examples/plot_ippo.py <run_dir> [<run_dir> ...]
Reads progress.jsonl (training) and periodic_eval.jsonl (evaluation),
saves two PNGs next to the first run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line; tolerate a trailing blank line."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _plot_training(run_dirs: list[Path], out: Path) -> None:
    """Env steps vs sampled agent-sum return (sanity that learning happens)."""
    plt.figure(figsize=(8, 5))
    for run in run_dirs:
        rows = _load_jsonl(run / "progress.jsonl")
        xs = [r["env_steps"] for r in rows if r.get("rllib_agent_sum_return_mean") is not None]
        ys = [r["rllib_agent_sum_return_mean"] for r in rows if r.get("rllib_agent_sum_return_mean") is not None]
        if xs:
            plt.plot(xs, ys, marker="o", label=run.name[:40])
    plt.xlabel("environment steps")
    plt.ylabel("sampled agent-sum return (train)")
    plt.title("IPPO training curve")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")


def _plot_eval(run_dirs: list[Path], out: Path) -> None:
    """Env steps vs deterministic F1 (truth metric) and team return."""
    fig, ax_f1 = plt.subplots(figsize=(8, 5))
    ax_ret = ax_f1.twinx()  # team return on the right y-axis
    for run in run_dirs:
        rows = _load_jsonl(run / "periodic_eval.jsonl")
        xs = [r["actual_env_step"] for r in rows]
        f1 = [r["summary"]["episode_task/musical_f1_mean"] for r in rows]
        ret = [r["summary"]["team_return_mean"] for r in rows]
        if xs:
            ax_f1.plot(xs, f1, marker="o", label=f"{run.name[:30]} F1")
            ax_ret.plot(xs, ret, marker="x", linestyle="--", alpha=0.5)
    ax_f1.set_xlabel("environment steps")
    ax_f1.set_ylabel("musical F1 (deterministic eval)")
    ax_ret.set_ylabel("team return (dashed)")
    ax_f1.set_ylim(0, 1)
    ax_f1.set_title("IPPO evaluation: F1 (truth) + team return")
    ax_f1.grid(True, alpha=0.3)
    ax_f1.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/plot_ippo.py <run_dir> [<run_dir> ...]")
        return 1
    run_dirs = [Path(p).expanduser().resolve() for p in sys.argv[1:]]
    base = run_dirs[0]
    _plot_training(run_dirs, base / "curve_training.png")
    _plot_eval(run_dirs, base / "curve_eval.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())