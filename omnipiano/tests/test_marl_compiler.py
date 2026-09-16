"""Fast regression tests for the pure MARL configuration compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnipiano.multiagent.compile.compiler import compile_experiment, compile_task
from omnipiano.multiagent.compile.schema import CONFIG_FIELDS, ResolvedTask


REGISTERED_ALGORITHMS = (
    "ippo",
    "ippo-rllib-module",
    "mappo",
    "mappo-own-critic",
)


def _compile(path: Path, *, algo: str | None = None):
    return compile_experiment(
        path,
        registered_algorithms=REGISTERED_ALGORITHMS,
        algo_override=algo,
    )


def test_canonical_config_compiles_without_changing_any_default() -> None:
    path = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_train_config_default.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    compiled = _compile(path)

    assert compiled.algorithm == raw["experiment"]["algo"] == "ippo"
    assert compiled.config_path == path.resolve()
    assert compiled.request.snapshot() == raw
    for section, fields in CONFIG_FIELDS.items():
        for json_key, destination in fields.items():
            assert compiled.values[destination] == raw[section][json_key]
    assert dict(compiled.smoke_test_overrides) == raw["smoke_test_overrides"]


def test_selected_algorithm_override_precedes_shared_defaults(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_train_config_default.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["ppo"]["num_epochs"] = 5
    raw["algorithm_overrides"]["mappo"] = {
        "ppo": {"num_epochs": 7},
    }
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    compiled = _compile(path, algo="mappo")

    assert compiled.algorithm == "mappo"
    assert compiled.values["algo"] == "mappo"
    assert compiled.values["num_epochs"] == 7
    assert compiled.request.snapshot()["experiment"]["algo"] == "ippo"


def test_legacy_config_without_reward_keeps_zero_penalty(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_train_config_default.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    del raw["reward"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    compiled = _compile(path)

    assert compiled.values["inter_agent_collision_penalty_coef"] == 0.0
    assert "reward" not in compiled.request.snapshot()


def test_unknown_field_fails_before_runtime_initialization(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_train_config_default.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["ppo"]["typo_num_epoch"] = 5
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields.*typo_num_epoch"):
        _compile(path)


def test_resolved_experiment_is_json_serializable_and_snapshot_isolated() -> None:
    path = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_train_config_default.json"
    compiled = _compile(path)

    serialized = json.loads(json.dumps(compiled.to_dict()))
    assert serialized["selected_algorithm"] == "ippo"
    assert serialized["source_config"] == str(path.resolve())

    copied = compiled.request.snapshot()
    copied["ppo"]["num_epochs"] = 999
    assert compiled.request.snapshot()["ppo"]["num_epochs"] != 999


def test_v2_task_inherits_training_defaults_and_resolves_ranges() -> None:
    path = Path(__file__).resolve().parents[1] / "multiagent" / "configs" / "marl_task_example.json"
    compiled = _compile(path)

    assert compiled.request.schema_version == 2
    assert compiled.values["total_steps"] == 10_000_000
    assert compiled.values["env_id"].startswith("OmniPiano-Custom-")
    assert compiled.task is not None
    assert [h.name for h in compiled.task.hands] == [
        "lh_b", "rh_b", "lh_t", "rh_t"
    ]
    secondo, primo = compiled.task.agents
    # User-facing 1..88 ranges become internal 0..87 exactly once.
    assert secondo.action_key_range == (0, 51)
    assert secondo.observation_key_range == (0, 59)
    assert primo.action_key_range == (36, 87)
    assert secondo.visible_teammate_hands == ("lh_t",)
    assert primo.visible_teammate_hands == ("rh_b",)
    assert ResolvedTask.from_dict(compiled.task.to_dict()) == compiled.task


def test_balanced_assignment_and_explicit_noncontiguous_assignment() -> None:
    balanced = compile_task({
        "song": "WinterWind", "num_hands": 5, "num_agents": 3,
        "assignment": "balanced",
    })
    assert [a.hand_ids for a in balanced.agents] == [(0, 1), (2, 3), (4,)]

    explicit = compile_task({
        "song": "WinterWind", "num_hands": 4, "num_agents": 2,
        "assignment": "explicit", "sustain_owner": "outer",
        "agents": [
            {"name": "outer", "hand_ids": [0, 3],
             "visible_teammate_hands": "all"},
            {"name": "inner", "hand_ids": [1, 2]},
        ],
    })
    assert explicit.agents[0].hand_ids == (0, 3)
    assert explicit.agents[0].visible_teammate_hands == ("rh_b", "lh_t")


@pytest.mark.parametrize("task,match", [
    ({"song": "WinterWind", "num_hands": 6}, "3, 4 or 5"),
    ({"song": "WinterWind", "num_hands": 4, "num_agents": 3,
      "assignment": "default"}, "conflicts"),
    ({"song": "WinterWind", "num_hands": 3, "num_agents": 2,
      "assignment": "explicit", "agents": [
          {"name": "a", "hand_ids": [0, 1]},
          {"name": "b", "hand_ids": [1, 2]},
      ]}, "unique zero-based"),
])
def test_invalid_tasks_fail_during_pure_compilation(task, match) -> None:
    with pytest.raises(ValueError, match=match):
        compile_task(task)
