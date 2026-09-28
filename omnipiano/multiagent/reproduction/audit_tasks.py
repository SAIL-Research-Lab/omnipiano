#!/usr/bin/env python3
"""Construct and audit the five frozen WinterWind SCHO tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from omnipiano.multiagent.train import _parse_args, _resolve_args
from omnipiano.multiagent.compile.schema import ResolvedTask
from omnipiano.multiagent.compile.environment import (
    make_parallel_from_task,
    prepare_task,
)
from omnipiano.multiagent.training.runtime import write_json


def own_obs(env, agent, value):
    value = np.asarray(value)
    layout = env.obs_layout(agent)
    start, end = layout.get("own", (0, value.size))
    return value[start:end]


def compare_observations(e0, e1, o0, o1):
    assert set(o0) == set(o1)
    for agent in o0:
        assert e0.observation_space(agent).contains(o0[agent]), agent
        assert e1.observation_space(agent).contains(o1[agent]), agent
        assert np.isfinite(o0[agent]).all(), agent
        assert np.isfinite(o1[agent]).all(), agent

        np.testing.assert_allclose(
            own_obs(e0, agent, o0[agent]),
            own_obs(e1, agent, o1[agent]),
            rtol=1e-6, atol=1e-6,
            err_msg=f"global-state mode changed actor observation: {agent}",
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument(
        "--output",
        default=None,
        help="Write the audit here instead of beside the frozen config suite.",
    )
    a = p.parse_args()

    runs_dir = Path(a.runs_dir).resolve()
    from robopianist import music

    midi_path = music._PIG_NAME_TO_FILE.get("EtudeOp25No11")
    if midi_path is None or not Path(midi_path).is_file():
        raise FileNotFoundError(
            "Install and preprocess PIG v1.2; EtudeOp25No11 is unavailable"
        )
    midi_path = Path(midi_path).resolve()
    report = {
        "dataset_identity": {
            "dataset": "Piano Fingering (PIG) dataset",
            "dataset_version": "1.2",
            "pig_piece_key": "EtudeOp25No11",
            "path": str(midi_path),
            "size_bytes": midi_path.stat().st_size,
            "sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
        },
        "interface_checks": {},
        "semantic_findings_from_source": {
            "action_key_range": (
                "per-owned-hand wrist-slider limits; not fingertip limits"
            ),
            "collision_penalty": (
                "-coefficient once per step if any cross-agent contact exists"
            ),
            "base_o_c_teammate_visibility_identical": True,
            "observation_changes_only_key_observation_ranges": True,
            "coupling_changes_only_action_ranges": True,
            "boundary_teammate_visibility": True,
        },
        "limitations": [
            "Random actions do not guarantee a real MuJoCo collision is triggered.",
            "This check does not measure maximum learned fingertip reach.",
            "This check does not execute a PPO learner update.",
        ],
    }
    total_action_dims = []

    paths = []
    for path in sorted(runs_dir.glob("*.json")):
        cfg = json.loads(path.read_text())
        if (
            cfg["experiment"]["algo"] == "ippo"
            and cfg["experiment"]["seed"] == 0
        ):
            paths.append((path, cfg))

    assert len(paths) == 5, f"Expected 5 IPPO seed-0 configs, got {len(paths)}"

    for path, cfg in paths:
        args = _parse_args([str(path)])
        spec = _resolve_args(args)
        assert spec.name == "ippo"

        snapshot = getattr(args, "_resolved_task", None)
        if snapshot is None:
            raise RuntimeError(
                "The config compiler did not expose _resolved_task."
            )

        resolved_task = prepare_task(ResolvedTask.from_dict(snapshot))
        resolved = resolved_task.to_dict()
        requested = cfg["task"]

        expected_agents = {
            x["name"]: sorted(x["hand_ids"])
            for x in requested["agents"]
        }

        assert requested["num_hands"] == 4
        assert len(resolved_task.hands) == 4
        assert [h.id for h in resolved_task.hands] == [0, 1, 2, 3]

        assert requested["num_agents"] == len(expected_agents)
        assert len(resolved_task.agents) == len(expected_agents)
        assert {
            x.name: list(x.hand_ids)
            for x in resolved_task.agents
        } == expected_agents

        assert resolved_task.song == requested["song"] == "WinterWind"
        assert resolved_task.base_env_name == (
            "RoboPianist-repertoire-150-EtudeOp25No11-v0"
        )

        hand_names = [h.name for h in resolved_task.hands]
        assert len(set(hand_names)) == 4

        compiled_agents = {
            agent.name: agent for agent in resolved_task.agents
        }

        for source in requested["agents"]:
            actual = compiled_agents[source["name"]]

            assert actual.action_key_range == tuple(
                key - 1 for key in source["action_key_range"]
            )
            assert actual.observation_key_range == tuple(
                key - 1 for key in source["observation_key_range"]
            )

            expected_visible = tuple(
                hand_names[h]
                for h in sorted(source["visible_teammate_hands"])
            )
            assert actual.visible_teammate_hands == expected_visible

        assert [
            agent.name
            for agent in resolved_task.agents
            if agent.is_sustain_owner
        ] == [requested["sustain_owner"]]

        prepared_hands = {
            hand["name"]: hand for hand in resolved_task.hand_specs
        }
        assert len(resolved_task.hand_specs) == 4
        assert set(prepared_hands) == set(hand_names)

        for agent in resolved_task.agents:
            for hand_id in agent.hand_ids:
                hand_name = hand_names[hand_id]
                assert tuple(
                    prepared_hands[hand_name]["key_range"]
                ) == agent.action_key_range

        envs = []
        try:
            for global_state in (False, True):
                envs.append(make_parallel_from_task(
                    resolved,
                    seed=123,
                    flatten_obs=True,
                    include_global_state=global_state,
                    inter_agent_collision_penalty_coef=0.1,
                ))
            e0, e1 = envs

            assert set(e0.possible_agents) == set(expected_agents)
            assert set(e1.possible_agents) == set(expected_agents)

            expected_runtime_assignment = {
                agent.name: tuple(hand_names[h] for h in agent.hand_ids)
                for agent in resolved_task.agents
            }

            for env in (e0, e1):
                assert set(env._agent_reaches) == set(expected_agents), (
                    "Reach cache does not match the current agent partition",
                    env._agent_reaches,
                )
                assert {
                    agent.name: tuple(agent.hand_names)
                    for agent in env._assignment.agents
                } == expected_runtime_assignment
                assert list(env._hand_action_slices) == hand_names

            action_dims = {}
            for index, agent in enumerate(e0.possible_agents):
                s0 = e0.action_space(agent)
                s1 = e1.action_space(agent)
                assert s0.shape == s1.shape
                np.testing.assert_array_equal(s0.low, s1.low)
                np.testing.assert_array_equal(s0.high, s1.high)
                s0.seed(1000 + index)
                action_dims[agent] = int(np.prod(s0.shape))

            o0, _ = e0.reset(seed=123)
            o1, _ = e1.reset(seed=123)
            compare_observations(e0, e1, o0, o1)

            own_dims = {
                agent: int(own_obs(e0, agent, o0[agent]).size)
                for agent in o0
            }

            for _ in range(a.steps):
                if not e0.agents:
                    break
                assert set(e0.agents) == set(e1.agents)

                actions = {
                    agent: e0.action_space(agent).sample()
                    for agent in e0.agents
                }
                o0, r0, term0, trunc0, _ = e0.step(actions)
                o1, r1, term1, trunc1, _ = e1.step({
                    k: v.copy() for k, v in actions.items()
                })

                assert term0 == term1
                assert trunc0 == trunc1
                assert set(r0) == set(r1)
                for agent in r0:
                    np.testing.assert_allclose(
                        r0[agent], r1[agent], rtol=1e-6, atol=1e-6
                    )

                rewards = np.asarray(list(r0.values()), dtype=float)
                assert np.isfinite(rewards).all()
                if rewards.size:
                    np.testing.assert_allclose(
                        rewards, np.full_like(rewards, rewards[0]),
                        rtol=1e-6, atol=1e-6,
                        err_msg="The environment is not broadcasting one team reward",
                    )

                compare_observations(e0, e1, o0, o1)

            total = sum(action_dims.values())
            total_action_dims.append(total)
            name = cfg["task"]["name"]
            report["interface_checks"][name] = {
                "passed": True,
                "resolved_task": resolved,
                "own_observation_dims": own_dims,
                "action_dims": action_dims,
                "joint_action_dim": total,
            }
            print(name, "OK", action_dims, "joint_dim=", total)

        finally:
            for env in envs:
                env.close()

    assert len(set(total_action_dims)) == 1, total_action_dims

    output = (
        Path(a.output).expanduser().resolve()
        if a.output is not None
        else runs_dir.parent / "environment_audit.json"
    )
    write_json(output, report)
    print("Saved:", output)


if __name__ == "__main__":
    main()
