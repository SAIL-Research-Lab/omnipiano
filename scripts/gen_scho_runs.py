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
        visible = [h for h in range(4) if h not in hands]
    return {
        "name": name,
        "hand_ids": list(hands),
        "action_key_range": list(action),
        "observation_key_range": list(observation),
        "visible_teammate_hands": list(visible),
    }


def task(name, agents):
    return {
        "name": f"scho-v1-winterwind-4h-{name}",
        "song": "WinterWind",
        "num_hands": 4,
        "num_agents": len(agents),
        "assignment": "explicit",
        "sustain_owner": "a0",
        "agents": agents,
    }


TASKS = {
    "base": task("base", [
        agent("a0", [0, 1], [1, 48], [1, 88]),
        agent("a1", [2, 3], [41, 88], [1, 88]),
    ]),
    "coupled_partial": task("coupled-partial", [
        agent("a0", [0, 1], [1, 60], [1, 48], []),
        agent("a1", [2, 3], [29, 88], [41, 88], []),
    ]),
    "heterogeneous": task("heterogeneous", [
        agent("a0", [0], [1, 48], [1, 88]),
        agent("a1", [1, 2, 3], [1, 88], [1, 88]),
    ]),
    "scaled": task("scaled", [
        agent("a0", [0], [1, 48], [1, 88]),
        agent("a1", [1], [1, 48], [1, 88]),
        agent("a2", [2], [41, 88], [1, 88]),
        agent("a3", [3], [41, 88], [1, 88]),
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
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument(
        "--run-root", default="/root/autodl-fs/omnipiano_runs"
    )
    a = p.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.suite_id):
        p.error("--suite-id must contain only letters, digits, '_' or '-'")
    if a.workers < 1:
        p.error("--workers must be >= 1")

    base = json.loads((ROOT / a.base).read_text(encoding="utf-8"))
    if base.get("schema_version") != 2 or "task" not in base:
        p.error("Use the supplied schema-v2 task template")

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_root = Path(a.run_root).resolve()

    preflight = a.phase == "preflight"
    total_steps = 20_000 if preflight else 10_000_000
    budget_tag = "20k-preflight" if preflight else "10M"

    for name, t in TASKS.items():
        validate_task(t)
        write_json(out / "tasks" / f"{name}.json", t)

    manifest = {
        "suite_id": a.suite_id,
        "phase": a.phase,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "task_hashes": {k: digest(v) for k, v in TASKS.items()},
        "semantic_checks": {
            "physical_hands_and_midi": "must verify after compilation",
            "range_fields_affect_runtime": "must verify in compiler/runtime",
            "partition_invariant_collision_reward": "not established",
            "heterogeneous_is_H_only": False,
        },
        "runs": [],
    }

    # Initial launch order:
    # each six-job block contains one setting and all algorithm/seed pairs.
    # Alternating the algorithm order balances IPPO/MAPPO across GPU slots.
    index = 0
    for task_index, (setting, t) in enumerate(TASKS.items()):
        for position in range(6):
            algo = ("ippo", "mappo")[(position + task_index) % 2]
            seed = (position // 2 + task_index) % 3

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
                ray_num_cpus=a.workers + 2,
                smoke_test=False,
                checkpoint_freq=(
                    10_000 if preflight else a.checkpoint_freq
                ),
            )
            if preflight:
                cfg["protocol"]["eval_freq"] = 10_000
                cfg["compute"]["log_every_iters"] = 1

            cfg["video"].update(
                enabled=False,
                record_final=False,
                wandb_upload=False,
            )

            tags = [
                a.suite_id, "scho-v1", setting, algo,
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

    assert index == 24
    assert len({
        (r["algo"], r["setting"], r["seed"])
        for r in manifest["runs"]
    }) == 24

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
    write_json(out / "all_72_runs_plan.json", plan)

    print(f"Generated 24 full configs: {out / 'runs'}")
    print(f"Generated 72-run plan:    {out / 'all_72_runs_plan.json'}")
    print("No training was launched.")
    print("Review semantic_checks before approving production runs.")


if __name__ == "__main__":
    main()