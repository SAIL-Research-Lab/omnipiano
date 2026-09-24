"""Plot completed custom runs; seed counts are explicit, no missing data filled."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_runs(root, group, source, smoke=False):
    records = []
    seen = set()
    for path in sorted((Path(root) / group).glob("*/cell.json")):
        cell = json.loads(path.read_text())
        status = path.parent / "status.json"
        if not cell.get("custom") or cell["smoke"] != smoke or not status.exists():
            continue
        state = json.loads(status.read_text())
        if state["status"] != "complete":
            continue
        key = cell["env_id"], cell["algorithm"], cell["seed"]
        if key in seen:
            raise ValueError("Duplicate task/algorithm/seed: select a separate --runs directory")
        seen.add(key)
        if source == "replay":
            with np.load(path.parent / "evaluations.npz", allow_pickle=False) as data:
                x = data["timesteps"].copy()
                series = {m: data[k].mean(axis=1) for m, k in
                          (("reward", "results"), ("cost", "ep_costs"), ("f1", "ep_f1"))}
        else:
            with (Path(state["save_dir"]) / "progress.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            x = np.array([float(r["TotalEnvSteps"]) for r in rows])
            series = {m: np.array([float(r[k]) for r in rows]) for m, k in
                      (("reward", "Metrics/EpRet"), ("cost", "Metrics/EpCost"))}
        if any(not np.isfinite(y).all() for y in series.values()):
            raise ValueError("Nonfinite curve values")
        records.append((cell, x, series))
    if not records:
        raise ValueError("No completed custom runs match this selection")
    signatures = {(c["code_hash"], json.dumps(c["runtime"], sort_keys=True),
                   c["steps"], c["eval_episodes"], c["evaluation_protocol"]) for c, _, _ in records}
    if len(signatures) != 1:
        raise ValueError("Separate different code/runtime/training/evaluation protocols before plotting")
    for env_id in {r[0]["env_id"] for r in records}:
        subset = [r for r in records if r[0]["env_id"] == env_id]
        if len({r[0]["midi_hash"] for r in subset}) != 1:
            raise ValueError("MIDI differs within a task")
        for algorithm in {r[0]["algorithm"] for r in subset}:
            configs = []
            for cell, _, _ in subset:
                if cell["algorithm"] == algorithm:
                    cfg = dict(cell["algorithm_config"])
                    cfg.pop("seed")
                    configs.append(json.dumps(cfg, sort_keys=True))
            if len(set(configs)) != 1:
                raise ValueError("Cannot aggregate seeds with different algorithm configurations")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--group", choices=("main", "hands", "budget", "extensions"), required=True)
    parser.add_argument("--out", type=Path, default=Path("safety_custom_figures"))
    parser.add_argument("--source", choices=("replay", "rollout"), default="replay")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = load_runs(args.runs, args.group, args.source, args.smoke_only)
    args.out.mkdir(parents=True, exist_ok=True)
    for env_id in sorted({r[0]["env_id"] for r in records}):
        subset = [r for r in records if r[0]["env_id"] == env_id]
        metrics = ("reward", "cost", "f1") if args.source == "replay" else ("reward", "cost")
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
        for ax, metric in zip(axes, metrics):
            for algorithm in sorted({r[0]["algorithm"] for r in subset}):
                runs = [r for r in subset if r[0]["algorithm"] == algorithm]
                x = runs[0][1]
                if any(not np.array_equal(x, r[1]) for r in runs):
                    raise ValueError("Unequal seed grids; no interpolation")
                y = np.stack([r[2][metric] for r in runs])
                mean = y.mean(axis=0)
                line, = ax.plot(x / 1e6, mean, label=f"{algorithm} (n={len(runs)})")
                if len(runs) > 1:
                    sd = y.std(axis=0, ddof=1)
                    ax.fill_between(x / 1e6, mean - sd, mean + sd, color=line.get_color(), alpha=.12)
            if metric == "cost":
                ax.axhline(subset[0][0]["task"]["budget"], color="black", linestyle=":", label="budget")
            ax.set_xlabel("Training steps (million)", fontsize=13)
            ax.set_ylabel(f"{args.source}: {metric}", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
        task = subset[0][0]["task"]
        prefix = "SMOKE TEST — " if args.smoke_only else "Completed runs only — "
        fig.suptitle(prefix + f"{task['hands']}H {task['song']}: {task['cost']['semantic']}/{task['cost']['setting']}"
                     + f"; budget={task['budget']:g}; mean ± seed SD", fontsize=13)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(args.out / f"{env_id}_{args.source}.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
