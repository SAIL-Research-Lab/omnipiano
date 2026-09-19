"""Main (8 panels), hands/budget (4 panels), strict three-seed aggregation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from omnipiano.safety.suite import ALGORITHMS, MAIN, HANDS, THRESHOLDS, BUDGET_TASK, cells


def load_records(root, group, source="replay"):
    records = []
    expected = {(t.env_id, a, s) for t, a, s, _ in cells(group)}
    for path in (Path(root) / group).glob("*/cell.json"):
        cell = json.loads(path.read_text())
        status = path.parent / "status.json"
        if cell["smoke"] or cell["steps"] != 5_000_000 or not status.exists():
            continue
        state = json.loads(status.read_text())
        if state["status"] != "complete":
            continue
        if cell["group"] != group or (cell["env_id"], cell["algorithm"], cell["seed"]) not in expected:
            continue
        if source == "replay":
            with np.load(path.parent / "evaluations.npz", allow_pickle=False) as data:
                x = data["timesteps"].copy()
                series = {m: data[k].mean(axis=1) for m, k in
                          (("reward", "results"), ("cost", "ep_costs"), ("f1", "ep_f1"))}
                if any(data[k].shape != (251, 10) for k in ("results", "ep_costs", "ep_f1")):
                    raise ValueError("Formal replay requires 251 checkpoints x 10 episodes")
            if not np.array_equal(x, np.arange(0, 5_000_001, 20_000)):
                raise ValueError("Wrong evaluation grid")
        else:
            with (Path(state["save_dir"]) / "progress.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            x = np.array([float(r["TotalEnvSteps"]) if "TotalEnvSteps" in r else float(r["Misc/TotalEnvSteps"]) for r in rows])
            series = {m: np.array([float(r[k]) for r in rows]) for m, k in
                      (("reward", "Metrics/EpRet"), ("cost", "Metrics/EpCost"))}
            if not np.array_equal(x, np.arange(20_000, 5_000_001, 20_000)):
                raise ValueError("Incomplete/unequal training rollout grid")
        if any(not np.isfinite(y).all() for y in series.values()):
            raise ValueError("Nonfinite curve values")
        records.append((cell, x, series))
    signatures = {(r[0]["code_hash"], json.dumps(r[0]["runtime"], sort_keys=True)) for r in records}
    if len(signatures) != 1:
        raise ValueError("Require one code/dependency version within each compared group")
    song_hashes = {}
    for cell, _, _ in records:
        song = cell["task"]["song"]
        if song in song_hashes and song_hashes[song] != cell["midi_hash"]:
            raise ValueError("Inconsistent MIDI version")
        song_hashes[song] = cell["midi_hash"]
    observed = [(c["env_id"], c["algorithm"], c["seed"]) for c, _, _ in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Missing or duplicate formal runs; no partial figures")
    return records


def mean_sd(runs, metric):
    if len(runs) != 3 or {r[0]["seed"] for r in runs} != {1, 2, 3}:
        raise ValueError("Require exactly training seeds 1, 2, 3")
    x = runs[0][1]
    if any(not np.array_equal(r[1], x) for r in runs):
        raise ValueError("Unequal grids; interpolation is not permitted")
    y = np.stack([r[2][metric] for r in runs])
    return x, y.mean(0), y.std(0, ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("main", "hands", "budget"), required=True)
    parser.add_argument("--runs", type=Path, default=Path("safety_runs"))
    parser.add_argument("--out", type=Path, default=Path("safety_figures"))
    parser.add_argument("--source", choices=("replay", "rollout"), default="replay")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = load_records(args.runs, args.group, args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    def draw(ax, task, algorithm, metric, label, color, style="-"):
        runs = [r for r in records if r[0]["env_id"] == task.env_id and r[0]["algorithm"] == algorithm]
        x, mean, sd = mean_sd(runs, metric)
        ax.plot(x / 1e6, mean, label=label, color=color, linestyle=style)
        ax.fill_between(x / 1e6, mean - sd, mean + sd, color=color, alpha=0.12)
    metrics = ("reward", "f1", "cost") if args.source == "replay" else ("reward", "cost")
    for metric in metrics:
        shape = (4, 2) if args.group == "main" else (2, 2)
        fig, axes = plt.subplots(*shape, figsize=(14, 3.5 * shape[0]), squeeze=False)
        for i, ax in enumerate(axes.flat):
            if args.group == "main":
                item = MAIN[i]
                for j, algorithm in enumerate(ALGORITHMS):
                    draw(ax, item, algorithm, metric, algorithm, f"C{j}")
                ax.set_title(f"{item.hands}H {item.song}: {item.cost.semantic}/{item.cost.setting}")
                if metric == "cost":
                    ax.axhline(item.budget, color="black", linestyle=":", label="budget")
            elif args.group == "hands":
                algorithm = ALGORITHMS[i + 1]
                for j, item in enumerate(HANDS):
                    draw(ax, item, algorithm, metric, f"{item.hands}H", f"C{j}")
                    draw(ax, item, "PPO", metric, f"{item.hands}H PPO", f"C{j}", "--")
                ax.set_title(algorithm)
                if metric == "cost":
                    ax.axhline(14.4, color="black", linestyle=":")
            else:
                algorithm = ALGORITHMS[i + 1]
                for j, item in enumerate(THRESHOLDS):
                    draw(ax, item, algorithm, metric, f"d={item.budget:g}", f"C{j}")
                    if metric == "cost":
                        ax.axhline(item.budget, color=f"C{j}", linestyle=":", alpha=.4)
                draw(ax, BUDGET_TASK, "PPO", metric, "PPO", "black", "--")
                ax.set_title(algorithm)
            ax.set_xlabel("Training environment steps (million)")
            ax.set_ylabel(f"{args.source}: {metric}")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("Mean ± sample SD across 3 training seeds")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(args.out / f"{args.group}_{args.source}_{metric}.{ext}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
