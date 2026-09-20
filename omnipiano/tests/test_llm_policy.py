import json

import numpy as np

from omnipiano.integrations.llm import (
    KeyframeConfig,
    KeyframeOptimizer,
    KeyframePolicy,
    LLMConfig,
    LLMPolicy,
)
from omnipiano.integrations.llm import keyframe_policy
from omnipiano.integrations.llm import keyframe_optimizer


class _ActionSpace:
    shape = (2,)
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)


def test_llm_policy_parses_json_action():
    policy = object.__new__(LLMPolicy)
    policy.action_space = _ActionSpace()
    action = policy._parse_action("```json\n[0.2, -0.4]\n```")
    np.testing.assert_allclose(action, [0.2, -0.4])


def test_llm_policy_passes_generation_options():
    policy = LLMPolicy(LLMConfig(
        action_space=_ActionSpace(), api_key="test",
        temperature=0.25, max_tokens=1234,
    ))

    assert policy.client.temperature == 0.25
    assert policy.client.max_tokens == 1234


def test_keyframe_policy_uses_environment_timestep(tmp_path, monkeypatch):
    path = tmp_path / "keyframes.json"
    path.write_text(
        json.dumps({"actions": [[0.1, 0.2], [0.3, 0.4]]}),
        encoding="utf-8",
    )
    task = type("Task", (), {"_t_idx": 0, "_notes": [[], []]})()
    composer_env = type("ComposerEnv", (), {"task": task})()
    monkeypatch.setattr(
        keyframe_policy,
        "get_composer_env_from_gym",
        lambda env: composer_env,
    )
    config = KeyframeConfig(
        env=object(),
        action_space=_ActionSpace(),
        path=str(path),
        api_key="test",
    )
    policy = KeyframePolicy(config)

    first, _ = policy.predict(None)
    task._t_idx = 1
    second, _ = policy.predict(None)
    task._t_idx = 0
    reset, _ = policy.predict(None)

    np.testing.assert_allclose(first, [0.1, 0.2])
    np.testing.assert_allclose(second, [0.3, 0.4])
    np.testing.assert_allclose(reset, first)


def test_keyframe_optimizer_runs_generated_program(tmp_path):
    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.action_space = _ActionSpace()
    optimizer._fk_joint_size = 2
    optimizer.path = tmp_path / "keyframes.json"
    source = (
        "import json, sys\n"
        "context = json.load(sys.stdin)\n"
        "print(json.dumps([[0.5, -0.5]] * context['num_steps']))\n"
    )
    optimizer._validate_program(source)
    optimizer._validate_program("import numpy as np\n")

    actions = optimizer._run_program(
        source, {
            "num_steps": 3, "action_size": 2,
            "action_low": [-2.0, -1.0], "action_high": [2.0, 1.0],
        }
    )
    np.testing.assert_allclose(actions, [[0.25, -0.5]] * 3)

    try:
        optimizer._validate_program("import os\n")
    except ValueError:
        pass
    else:
        raise AssertionError("generated controller must not import network/filesystem modules")

def test_keyframe_action_metadata_describes_targets_and_pedal():
    joint = type("Joint", (), {
        "name": "forearm_ty",
        "full_identifier": "hand/forearm_ty",
        "type": "slide",
    })()
    actuator = type("Actuator", (), {
        "full_identifier": "hand/forearm_ty", "joint": joint
    })()
    hand = type("Hand", (), {"joints": [joint], "actuators": [actuator]})()
    task = type("Task", (), {"hands": [hand]})()
    metadata = KeyframeOptimizer._action_metadata(task)

    assert metadata[0]["target"] == "hand/forearm_ty"
    assert metadata[0]["joint_type"] == "slide"
    assert set(metadata[0]) == {"target", "joint_type"}
    assert metadata[1] == {
        "target": "piano sustain pedal",
        "joint_type": "pedal",
    }


def test_keyframe_fk_ranges_use_compiled_physics_values():
    joints = [
        type("Joint", (), {"range": None})(),
        type("Joint", (), {"range": [-1.0, 1.0]})(),
    ]
    compiled_ranges = np.array([[-0.5, 0.25], [-1.0, 1.0]])
    physics = type("Physics", (), {
        "bind": lambda self, targets: type(
            "Bound", (), {"range": compiled_ranges}
        )()
    })()

    ranges = KeyframeOptimizer._joint_ranges(physics, joints)

    np.testing.assert_allclose(ranges, compiled_ranges)


def test_forward_kinematics_starts_each_query_from_fixed_state(monkeypatch):
    actuators, fingertips, joints = object(), object(), object()

    class Physics:
        def __init__(self):
            self.state = np.array([99.0])
            self.data = type("Data", (), {})()
            self.data.ctrl = np.array([9.0])
            self.forward_calls = 0

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = state.copy()

        def reset_context(self):
            class Context:
                def __enter__(inner):
                    return inner

                def __exit__(inner, *_):
                    pass
            return Context()

        def bind(self, target):
            physics = self

            class Bound:
                @property
                def ctrl(self):
                    return physics.data.ctrl

                @ctrl.setter
                def ctrl(self, value):
                    physics.data.ctrl[:] = value

                @property
                def xpos(self):
                    return np.array([[physics.state[0]]])

                @property
                def qpos(self):
                    return physics.state.copy()

                @qpos.setter
                def qpos(self, value):
                    physics.state[:] = value
            return Bound()

        def forward(self):
            self.forward_calls += 1

    physics = Physics()
    spec = type("Spec", (), {
        "minimum": np.array([0.0, 0.0]),
        "maximum": np.array([2.0, 1.0]),
    })()
    hand = type("Hand", (), {
        "name": "hand", "actuators": actuators,
        "fingertip_sites": fingertips, "joints": [joints],
        "action_spec": lambda self, physics: type("Shape", (), {"shape": (1,)})(),
    })()
    task = type("Task", (), {
        "hands": [hand], "action_spec": lambda self, physics: spec,
    })()
    monkeypatch.setattr(
        keyframe_optimizer, "get_composer_env_from_gym",
        lambda env: type("Composer", (), {"task": task, "physics": physics})(),
    )
    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.env = object()
    optimizer.action_space = _ActionSpace()
    optimizer._fk_joint_size = 1
    optimizer._fk_state = np.array([0.0])
    optimizer._fk_ctrl = np.array([0.0])

    result = optimizer._forward_kinematics([[1.0], [1.0]])

    assert result[0]["fingertip_positions"] == result[1]["fingertip_positions"]
    assert physics.forward_calls == 2
    np.testing.assert_allclose(physics.state, [99.0])
    np.testing.assert_allclose(physics.data.ctrl, [9.0])


def test_keyframe_tools_and_error_ranges():
    optimizer = object.__new__(KeyframeOptimizer)
    optimizer._robot_context = {
        "robot_model": "Shadow Hand", "hands": ["rh_shadow_hand"],
        "fingertip_names": {"rh_shadow_hand": ["thdistal_site"]},
        "action_names": ["rh_shadow_hand/forearm_tx", "sustain"],
        "action_low": [-1.0, 0.0], "action_high": [1.0, 1.0],
        "action_metadata": [
            {"target": "rh_shadow_hand/forearm_tx", "joint_type": "slide"},
            {"target": "piano sustain pedal", "joint_type": "pedal"},
        ],
        "fk_joint_names": ["rh_shadow_hand/forearm_tx"],
        "fk_joint_low": [-1.0], "fk_joint_high": [1.0],
    }
    optimizer.trajectories = {1: [{"step": step, "error": step in {0, 1, 4}} for step in range(5)]}
    optimizer._fk_joint_size = 2
    optimizer._fk_joint_low = np.array([-1.0, -1.0])
    optimizer._fk_joint_high = np.array([1.0, 1.0])
    optimizer._forward_kinematics = lambda queries: queries

    assert optimizer._tool_result("get_practice_trajectory", {"attempt": 1, "start_step": 1, "end_step": 4}) == optimizer.trajectories[1][1:]
    assert optimizer._tool_result("fk_query", {"joint_positions": [[0.0, 1.0]]}) == [[0.0, 1.0]]
    batch = [[0.0, 0.0]] * 8
    assert optimizer._tool_result("fk_query", {"joint_positions": batch}) == batch
    naming = optimizer._tool_result("get_robot_reference", {"section": "naming"})
    assert naming["action_names"] == ["rh_shadow_hand/forearm_tx", "sustain"]
    forearm = optimizer._tool_result("get_robot_reference", {"section": "forearm"})
    assert forearm["active_forearm_actuators"]["rh_shadow_hand/forearm_tx"]["hand_local_axis"] == [-1, 0, 0]
    assert KeyframeOptimizer._error_ranges(optimizer.trajectories[1]) == [[0, 1], [4, 4]]
    for arguments in ({"attempt": 2, "start_step": 0, "end_step": 0}, {"attempt": 1, "start_step": 4, "end_step": 5}):
        try:
            optimizer._tool_result("get_practice_trajectory", arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("trajectory tool must reject unavailable ranges")
    for positions in ([[0.0]], [[float("nan"), 0.0]], [[2.0, 0.0]]):
        try:
            optimizer._tool_result("fk_query", {"joint_positions": positions})
        except ValueError:
            pass
        else:
            raise AssertionError("FK tool must validate inputs")


def test_keyframe_tool_schemas_are_closed():
    tools = {tool["function"]["name"]: tool["function"] for tool in KeyframeOptimizer._tools()}

    assert set(tools) == {"get_robot_reference", "get_practice_trajectory", "fk_query"}
    assert tools["get_practice_trajectory"]["parameters"]["additionalProperties"] is False
    assert tools["fk_query"]["parameters"]["properties"]["joint_positions"]["maxItems"] == 8
    assert [tool["function"]["name"] for tool in KeyframeOptimizer._tools(
        include_reference=False, include_fk=False, include_trajectory=True,
    )] == ["get_practice_trajectory"]
    assert [tool["function"]["name"] for tool in KeyframeOptimizer._tools(
        include_fk=True, include_trajectory=False,
    )] == ["get_robot_reference", "fk_query"]


def test_keyframe_agent_keeps_tool_calls_in_one_message_history(tmp_path):
    call = _tool_call(
        "call_1", "get_practice_trajectory",
        {"attempt": 1, "start_step": 0, "end_step": 0},
    )

    class Message:
        def __init__(self, content, calls=()):
            self.content, self.tool_calls = content, calls

        def model_dump(self, **_):
            result = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                result["tool_calls"] = [{
                    "id": call.id, "type": "function",
                    "function": {"name": call.function.name,
                                 "arguments": call.function.arguments},
                } for call in self.tool_calls]
            return result

    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.path = tmp_path / "keyframes.json"
    optimizer.trajectories = {1: [{"step": 0, "error": True}]}
    optimizer.client = type("Client", (), {
        "model": "test", "thinking": False,
        "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "chat": lambda _, messages, tools: responses.pop(0),
    })()
    responses = [Message(None, [call]), Message("```python\nprint([])\n```")]
    messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "context"}]

    _, source = optimizer._request_program(messages)

    assert source == "print([])\n"
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    saved = json.loads((tmp_path / "keyframes_conversation.json").read_text())
    assert saved["messages"] == messages
    assert saved["token_usage"]["total_tokens"] == 3


def _tool_call(call_id, name, arguments):
    return type("Call", (), {
        "id": call_id,
        "function": type("Function", (), {
            "name": name,
            "arguments": json.dumps(arguments),
        })(),
    })()


def test_keyframe_agent_limits_fk_calls_and_resets_budget(tmp_path):
    class Message:
        def __init__(self, content=None, calls=()):
            self.content, self.tool_calls = content, calls

        def model_dump(self, **_):
            result = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                result["tool_calls"] = [{
                    "id": call.id, "type": "function",
                    "function": {"name": call.function.name,
                                 "arguments": call.function.arguments},
                } for call in self.tool_calls]
            return result

    responses = [
        Message(calls=[_tool_call(str(index), "fk_query", {"joint_positions": [[0.0]]})])
        for index in range(7)
    ] + [Message("```python\nprint([])\n```")]
    seen_tools, executed = [], []

    def chat(_, messages, tools):
        seen_tools.append([tool["function"]["name"] for tool in tools])
        return responses.pop(0)

    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.path = tmp_path / "keyframes.json"
    optimizer.trajectories = {}
    optimizer.client = type("Client", (), {
        "model": "test", "thinking": False,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "chat": chat,
    })()
    optimizer._tool_result = lambda name, arguments: executed.append(arguments) or {"ok": True}
    messages = [{"role": "user", "content": "context"}]

    optimizer._request_program(messages)

    assert len(executed) == 6
    assert all("fk_query" in tools for tools in seen_tools[:6])
    assert all("fk_query" not in tools for tools in seen_tools[6:])
    assert any("limit of 6 calls reached" in (message.get("content") or "") for message in messages)

    responses.append(Message("```python\nprint([])\n```"))
    optimizer._request_program(messages)
    assert seen_tools[-1] == ["get_robot_reference", "fk_query"]
    assert not (tmp_path / "keyframes_conversation.json.tmp").exists()


def test_keyframe_agent_removes_each_tool_after_its_budget(tmp_path):
    class Message:
        def __init__(self, content=None, calls=()):
            self.content, self.tool_calls = content, calls

        def model_dump(self, **_):
            return {"role": "assistant", "content": self.content}

    for name, limit in keyframe_optimizer._TOOL_CALL_LIMITS.items():
        arguments = {"section": "overview"} if name == "get_robot_reference" else (
            {"joint_positions": [[0.0]]} if name == "fk_query" else
            {"attempt": 1, "start_step": 0, "end_step": 0}
        )
        responses = [Message(calls=[_tool_call(str(index), name, arguments)])
                     for index in range(limit)] + [Message("```python\nprint([])\n```")]
        seen_tools = []

        def chat(_, messages, tools):
            seen_tools.append([tool["function"]["name"] for tool in tools])
            return responses.pop(0)

        optimizer = object.__new__(KeyframeOptimizer)
        optimizer.path = tmp_path / f"{name}.json"
        optimizer.trajectories = {1: [{}]}
        optimizer.client = type("Client", (), {
            "model": "test", "thinking": False,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "chat": chat,
        })()
        optimizer._tool_result = lambda *_: {"ok": True}
        messages = [{"role": "user", "content": "context"}]

        optimizer._request_program(messages)

        assert all(name in tools for tools in seen_tools[:-1])
        assert name not in seen_tools[-1]
        assert not any("All available tool budgets are exhausted" in (message.get("content") or "")
                       for message in messages)


def test_keyframe_agent_requires_output_only_after_all_tools_are_exhausted(tmp_path):
    class Message:
        def __init__(self, content=None, calls=()):
            self.content, self.tool_calls = content, calls

        def model_dump(self, **_):
            return {"role": "assistant", "content": self.content}

    responses = [
        Message(calls=[_tool_call(str(index), "get_robot_reference", {"section": "overview"})])
        for index in range(keyframe_optimizer._MAX_ROBOT_REFERENCE_CALLS)
    ] + [
        Message(calls=[_tool_call(str(index), "fk_query", {"joint_positions": [[0.0]]})])
        for index in range(keyframe_optimizer._MAX_FK_CALLS)
    ] + [Message("```python\nprint([])\n```")]
    seen_tools = []

    def chat(_, messages, tools):
        seen_tools.append([tool["function"]["name"] for tool in tools])
        return responses.pop(0)

    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.path = tmp_path / "all_tools.json"
    optimizer.trajectories = {}
    optimizer.client = type("Client", (), {
        "model": "test", "thinking": False,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "chat": chat,
    })()
    optimizer._tool_result = lambda *_: {"ok": True}
    messages = [{"role": "user", "content": "context"}]

    optimizer._request_program(messages)

    assert seen_tools[-1] == []
    assert any(
        "All available tool budgets are exhausted" in (message.get("content") or "")
        and "immediately return the complete play.py" in (message.get("content") or "")
        for message in messages
    )


def test_keyframe_fingertip_positions_are_dynamic():
    sites = [type("Site", (), {"name": f"tip_{index}"})() for index in range(12)]
    hand = type("Hand", (), {"name": "many_fingered_hand", "fingertip_sites": sites})()
    positions = np.arange(36).reshape(12, 3)
    physics = type("Physics", (), {
        "bind": lambda self, target: type("Bound", (), {"xpos": positions})()
    })()

    result = KeyframeOptimizer._fingertip_positions(physics, [hand])

    assert len(result["many_fingered_hand"]) == 12
    assert result["many_fingered_hand"]["tip_11"] == [33, 34, 35]


def test_keyframe_names_only_nonzero_physical_actions():
    result = KeyframeOptimizer._named_nonzero_actions(
        ["joint_a", "joint_b", "sustain"], [0.0, 0.25, -0.5]
    )

    assert result == {"joint_b": 0.25, "sustain": -0.5}


def test_keyframe_prompt_has_five_compact_sections_and_omits_active_fingers():
    optimizer = object.__new__(KeyframeOptimizer)
    prompt = optimizer._prompt({
        "environment_id": "OmniPiano-Test-ThreeHand-StaticPartition-v0",
        "hand_count": 3,
        "static_partition": True,
        "hand_partitions": [{"hand": "rh_a", "key_range": [0, 28], "y_range": [-0.6, -0.2]}],
        "num_steps": 2,
        "action_size": 3,
        "control_timestep": 0.05,
        "fk_joint_names": ["joint"],
        "score": [{"key": 1, "onset": 0.0, "offset": 0.1}],
    })

    assert [line for line in prompt.splitlines() if line.startswith("**")] == [
        "**Task:**", "**Score:**", "**Rules:**", "**Tools:**", "**Context:**",
    ]
    assert "active_fingers" not in prompt
    assert "correct piano keys at their scheduled times" in prompt
    assert "Avoid both wrong and missed key presses" in prompt
    assert "maximize episode task F1" in prompt
    assert "zero-based piano-key index (0-87)" in prompt
    assert "finger indexes finger_names" in prompt
    assert "finger_names[finger]" in prompt
    assert "right thumb, right index" not in prompt
    assert "only authoritative source for actuator and joint names" in prompt
    assert "Never construct names by adding a hand prefix, rh_/lh_, or a suffix" in prompt
    assert "look up action indices using complete identifiers" in prompt
    assert "action_metadata and action_names are aligned by index" in prompt
    assert "Read JSON context from sys.stdin" in prompt
    assert "physical actuator targets in action_names order" in prompt
    assert "normalized joint target" not in prompt
    assert "do not normalize them" in prompt
    assert "fk_query returns joint positions, not an action row" in prompt
    assert "match direct joint actuators by action_metadata target name" in prompt
    assert "tune tendon actuators separately" in prompt
    assert "coordinate differences between poses are not additive offsets" in prompt
    assert "get_practice_trajectory" in prompt
    assert "available only after a practice attempt completes" in prompt
    assert "get_robot_reference" in prompt
    assert "at most 6 times" in prompt
    assert "Batch up to 8 carefully chosen poses" in prompt
    assert "substantive, diagnostics-driven change" in prompt
    assert '"environment_id":"OmniPiano-Test-ThreeHand-StaticPartition-v0"' in prompt
    assert '"hand_count":3' in prompt
    assert '"static_partition":true' in prompt
    assert '"hand_partitions"' in prompt
    assert '"num_steps":2' in prompt
    assert '"num_steps": 2' not in prompt


def test_keyframe_save_records_history_and_token_usage(tmp_path):
    optimizer = object.__new__(KeyframeOptimizer)
    optimizer.path = tmp_path / "keyframes_env.json"
    optimizer.program_path = tmp_path / "play_env.py"
    optimizer.practice_seed = 7
    optimizer.client = type("Client", (), {
        "model": "test-model",
        "token_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    })()
    history = [{
        "attempt": 1, "f1": 0.2, "precision": 0.3, "recall": 0.1,
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 4,
                        "total_tokens": 14},
    }]

    optimizer._save(
        {"control_timestep": 0.05}, history, history[0], "print([])\n",
        np.zeros((1, 2)), 1,
    )

    summary = json.loads((tmp_path / "optimization_summary.json").read_text())
    assert summary["practice_attempts"] == history
    assert summary["token_usage"]["total_tokens"] == 14
