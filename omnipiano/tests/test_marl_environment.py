"""Tests for resolved-task preparation and registered-env translation."""

from pathlib import Path

from omnipiano.multiagent.compile.compiler import compile_experiment
from omnipiano.multiagent.compile.environment import (
    prepare_task,
    resolve_registered_task,
)
from omnipiano.multiagent.compile.schema import ResolvedTask


ALGOS = (
    "ippo", "ippo-rllib-module", "mappo", "mappo-own-critic",
    "happo", "facmac",
)


def test_prepared_task_snapshot_contains_physics_inputs() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "multiagent"
        / "configs"
        / "marl_task_example.json"
    )
    compiled = compile_experiment(path, registered_algorithms=ALGOS)
    prepared = prepare_task(compiled.task)
    restored = ResolvedTask.from_dict(prepared.to_dict())

    assert restored == prepared
    assert len(prepared.hand_specs) == 4
    assert prepared.env_config["disable_fingering_reward"] is True
    assert [tuple(h["key_range"]) for h in prepared.hand_specs] == [
        (0, 51), (0, 51), (36, 87), (36, 87)
    ]


def test_registered_four_hand_env_resolves_to_same_explicit_contract() -> None:
    env_id = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
    task = resolve_registered_task(env_id)

    assert task.legacy_env_id == env_id
    assert task.song == "WinterWind"
    assert task.layout == "FourHand"
    assert [h.name for h in task.hands] == ["lh_b", "rh_b", "lh_t", "rh_t"]
    assert [h.bucket_key_range for h in task.hands] == [
        (0, 21), (22, 43), (44, 65), (66, 87)
    ]
    assert [a.name for a in task.agents] == ["secondo", "primo"]
    assert [a.hand_ids for a in task.agents] == [(0, 1), (2, 3)]
    assert [a.action_key_range for a in task.agents] == [(0, 43), (44, 87)]
    assert [a.observation_key_range for a in task.agents] == [None, None]
    assert [a.visible_teammate_hands for a in task.agents] == [
        ("lh_t",), ("rh_b",)
    ]
    assert [tuple(h["key_range"]) for h in task.hand_specs] == [
        (0, 43), (0, 43), (44, 87), (44, 87)
    ]
    assert task.env_config is not None
    assert task.task_config is not None
    assert task.robust_config is not None


def test_default_assignment_runtime_objects_are_derived_from_presets() -> None:
    from omnipiano.multiagent.compile.env_runtime.topology import (
        AGENT_ASSIGNMENTS,
    )
    from omnipiano.multiagent.compile.presets import DEFAULT_ASSIGNMENTS, LAYOUTS

    for num_hands, groups in DEFAULT_ASSIGNMENTS.items():
        morphology, hand_names, _ = LAYOUTS[num_hands]
        actual = AGENT_ASSIGNMENTS[morphology]
        assert tuple(
            (agent.name, tuple(hand_names.index(h) for h in agent.hand_names))
            for agent in actual.agents
        ) == groups
