#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALL_ALGOS = ("ippo", "mappo", "happo", "mat", "facmac", "masac")
RUNNABLE_ALGOS = ("ippo", "mappo", "facmac")


def digest(value):
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def agent(name, hands, action, observation, visible=None):
    if visible is None:
        visible = "boundary"
    return {
        "name": name,
        "hand_ids": list(hands),
        "action_key_range": list(action),
        "observation_key_range": list(observation),
        "visible_teammate_hands": (
            list(visible) if not isinstance(visible, str) else visible
        ),
    }


def task(name, agents):
    return {
        "name": f"scho-v2-winterwind-4h-{name}",
        "song": "WinterWind",
        "num_hands": 4,
        "num_agents": len(agents),
        "assignment": "explicit",
        "sustain_owner": "agent_1",
        "agents": agents,
    }


TASKS = {
    "base": task("base", [
        agent("agent_1", [0, 1], [1, 44], [1, 49], [2]),
        agent("agent_2", [2, 3], [45, 88], [40, 88], [1]),
    ]),
    "observability": task("observability", [
        agent("agent_1", [0, 1], [1, 44], [1, 70], [2]),
        agent("agent_2", [2, 3], [45, 88], [19, 88], [1]),
    ]),
    "coupling": task("coupling", [
        agent("agent_1", [0, 1], [1, 70], [1, 49], [2]),
        agent("agent_2", [2, 3], [19, 88], [40, 88], [1]),
    ]),
    "heterogeneous": task("heterogeneous", [
        agent("agent_1", [0], [1, 22], [1, 27], [1]),
        agent("agent_2", [1, 2, 3], [23, 88], [18, 88], [0]),
    ]),
    "scalability": task("scalability", [
        agent("agent_1", [0], [1, 22], [1, 27], [1]),
        agent("agent_2", [1], [23, 44], [18, 49], [0, 2]),
        agent("agent_3", [2], [45, 66], [40, 71], [1, 3]),
        agent("agent_4", [3], [67, 88], [62, 88], [2]),
    ]),
}


def validate_task(t):
    agents = t["agents"]
    names = [a["name"] for a in agents]
    hands = [h for a in agents for h in a["hand_ids"]]

    assert t["num_hands"] == 4
    assert t["num_agents"] == len(agents)
    assert len(set(names)) == len(names)
    assert sorted(hands) == [0, 1, 2, 3]
    assert t["sustain_owner"] in names

    for a in agents:
        for field in ("action_key_range", "observation_key_range"):
            lo, hi = a[field]
            assert 1 <= lo <= hi <= 88, (a["name"], field, lo, hi)
        assert not set(a["hand_ids"]) & set(a["visible_teammate_hands"])
        assert set(a["visible_teammate_hands"]) <= set(range(4))


def git_output(*args):
    p = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True, capture_output=True,
    )
    return p.stdout.strip() if p.returncode == 0 else "unavailable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default="omnipiano/multiagent/configs/marl_task_example.json",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--suite-id", required=True)
    p.add_argument("--phase", choices=("preflight", "train"), required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument(
        "--algos", nargs="+", choices=RUNNABLE_ALGOS,
        default=("ippo", "mappo"),
        help="Algorithms to materialize; defaults preserve the original suite.",
    )
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--run-root", default="/root/autodl-fs/omnipiano_runs"
    )
    a = p.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.suite_id):
        p.error("--suite-id must contain only letters, digits, '_' or '-'")
    if a.workers < 1:
        p.error("--workers must be >= 1")
    if a.start_index < 0:
        p.error("--start-index must be non-negative")

    base = json.loads((ROOT / a.base).read_text(encoding="utf-8"))
    if base.get("schema_version") != 2 or "task" not in base:
        p.error("Use the supplied schema-v2 task template")

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_root = Path(a.run_root).resolve()

    preflight = a.phase == "preflight"
    total_steps = 20_000 if preflight else 10_000_000
    budget_tag = "20k-preflight" if preflight else "10M"
    selected_algos = tuple(dict.fromkeys(a.algos))
    facmac_defaults = json.loads(
        (
            ROOT
            / "omnipiano/multiagent/configs/marl_train_config_default.json"
        ).read_text(encoding="utf-8")
    )["algorithm_overrides"]["facmac"]["native"]

    for name, t in TASKS.items():
        validate_task(t)
        write_json(out / "tasks" / f"{name}.json", t)

    manifest = {
        "suite_id": a.suite_id,
        "phase": a.phase,
        "selected_algorithms": list(selected_algos),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "task_hashes": {k: digest(v) for k, v in TASKS.items()},
        "semantic_checks": {
            "physical_hands_and_midi": "must verify after compilation",
            "range_fields_affect_runtime": "must verify in compiler/runtime",
            "base_o_c_teammate_visibility_identical": True,
            "observation_changes_only_key_observation_ranges": True,
            "coupling_changes_only_action_ranges": True,
            "boundary_teammate_visibility": True,
        },
        "runs": [],
    }

    # Initial launch order covers every algorithm/task pair before queuing the
    # second and third seeds. This maximizes task coverage under a constrained
    # process or GPU budget while keeping matched seeds in the same suite.
    index = a.start_index
    pairs = [
        (setting, selected_algos[(task_index + offset) % len(selected_algos)])
        for offset in range(len(selected_algos))
        for task_index, setting in enumerate(TASKS)
    ]
    assert len(set(pairs)) == len(TASKS) * len(selected_algos)
    for seed in (0, 1, 2):
        for setting, algo in pairs:
            t = TASKS[setting]

            cfg = copy.deepcopy(base)
            name = (
                f"{a.suite_id}_{setting}_{algo}_s{seed}_c01_{budget_tag}"
            )

            cfg["schema_version"] = 2
            cfg["task"] = copy.deepcopy(t)
            cfg["experiment"].update(
                algo=algo,
                env_id=None,
                seed=seed,
                run_dir=str(run_root / a.suite_id / a.phase / name),
            )
            cfg["protocol"]["total_steps"] = total_steps
            cfg["reward"]["inter_agent_collision_penalty_coef"] = 0.1

            cfg["compute"].update(
                num_workers=a.workers,
                num_cpus_per_env_runner=1,
                num_learners=1,
                num_gpus_per_learner=1.0,
                ray_num_cpus=(None if algo == "facmac" else a.workers + 2),
                smoke_test=False,
                checkpoint_freq=(
                    10_000 if preflight else a.checkpoint_freq
                ),
            )
            if algo == "facmac":
                # Freeze the complete native recipe into every run instead of
                # relying on whichever defaults happen to be installed later.
                cfg["native"] = copy.deepcopy(facmac_defaults)
                # FACMAC updates every 24 environment steps.  Logging every
                # update would create ~417k W&B/JSONL rows per 10M run while
                # adding no training information.  25 update blocks still
                # retain ~16.7k learner points and every 50k evaluation point.
                cfg["compute"]["log_every_iters"] = 25
            if preflight:
                cfg["protocol"]["eval_freq"] = 10_000
                cfg["compute"]["log_every_iters"] = 1

            cfg["video"].update(
                enabled=True,
                freq=500_000,
                record_final=True,
                wandb_upload=True,
            )

            tags = [
                a.suite_id, "scho-v2", setting, algo,
                "WinterWind", "4hands", f"{t['num_agents']}agents",
                f"seed{seed}", "collision-0.1", budget_tag,
                f"workers{a.workers}",
            ]
            cfg["wandb"].update(
                name=name,
                group=f"{a.suite_id}_{setting}_{algo}_{budget_tag}",
                tags=",".join(tags),  # Matches the supplied JSON schema.
                notes=(
                    f"task_sha256={digest(t)}; "
                    f"train_env_steps={total_steps}; "
                    "canonical configuration; see suite manifest for "
                    "axis-isolation and reward-scope checks."
                ),
            )
            if preflight:
                cfg["wandb"]["mode"] = "offline"

            cfg["description"] = cfg["wandb"]["notes"]
            filename = f"{index:02d}_{name}.json"
            write_json(out / "runs" / filename, cfg)

            manifest["runs"].append({
                "file": f"runs/{filename}",
                "algo": algo,
                "setting": setting,
                "seed": seed,
                "task_sha256": digest(t),
                "config_sha256": digest(cfg),
            })
            index += 1

    expected_runs = len(TASKS) * len(selected_algos) * 3
    assert index == a.start_index + expected_runs
    assert len({
        (r["algo"], r["setting"], r["seed"])
        for r in manifest["runs"]
    }) == expected_runs

    write_json(out / "manifest.json", manifest)

    plan = [
        {
            "algo": algo,
            "setting": setting,
            "seed": seed,
            "task_sha256": digest(t),
            "total_train_env_steps": 10_000_000,
            "collision_coef": 0.1,
            "implementation": (
                "existing backend; needs suite validation"
                if algo in ("ippo", "mappo")
                else "not released for this suite"
            ),
        }
        for algo in ALL_ALGOS
        for setting, t in TASKS.items()
        for seed in (0, 1, 2)
    ]
    write_json(out / "all_90_runs_plan.json", plan)

    print(f"Generated {expected_runs} full configs: {out / 'runs'}")
    print(f"Generated 90-run plan:    {out / 'all_90_runs_plan.json'}")
    print("No training was launched.")
    print("Review semantic_checks before approving production runs.")


if __name__ == "__main__":
    main()
