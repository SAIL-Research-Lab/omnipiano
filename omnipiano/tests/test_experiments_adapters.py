"""Regression tests for side-effect-free unified backend binding."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from omnipiano.experiments import (
    AdapterError,
    bind_plan,
    get_track_capability,
    load_request,
    parse_request,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "omnipiano" / "experiments" / "examples"


def _declared_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }


@pytest.mark.parametrize(
    ("track", "backend", "prefix"),
    [
        ("standard", "stable_baselines3", ("python", "examples/run_sb3_baseline.py")),
        ("robust", "torchrl", ("python", "examples/run_torchrl_robust.py")),
        ("safety", "omnisafe", ("python", "-m", "omnipiano.safety.run")),
        ("marl", "omnipiano_marl", ("python", "-m", "omnipiano.multiagent.train")),
    ],
)
def test_examples_bind_all_seeds_without_writing(
    track: str,
    backend: str,
    prefix: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle = bind_plan(load_request(EXAMPLES / f"{track}.json"))

    assert len(bundle.runs) == 3
    assert [run.seed for run in bundle.runs] == [0, 1, 2]
    assert all(run.launch_status == "bound" for run in bundle.runs)
    assert all(run.backend == backend for run in bundle.runs)
    assert all(run.command[: len(prefix)] == prefix for run in bundle.runs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("track", ["standard", "robust", "safety"])
def test_every_advertised_single_agent_algorithm_binds(track: str) -> None:
    base = json.loads((EXAMPLES / f"{track}.json").read_text(encoding="utf-8"))
    for capability in get_track_capability(track).algorithms:
        if not capability.selectable:
            continue
        document = json.loads(json.dumps(base))
        document["algorithm"] = {"name": capability.name}
        bundle = bind_plan(parse_request(document))
        assert all(run.launch_status == "bound" for run in bundle.runs)
        assert all(capability.name in run.command for run in bundle.runs)


def test_every_advertised_multiagent_algorithm_binds_on_a_compatible_task() -> None:
    registered = json.loads((EXAMPLES / "marl.json").read_text(encoding="utf-8"))
    for capability in get_track_capability("marl").algorithms:
        if not capability.selectable or capability.name == "ppo-monolithic":
            continue
        document = json.loads(json.dumps(registered))
        document["algorithm"] = {"name": capability.name}
        bundle = bind_plan(parse_request(document))
        assert all(run.backend == "omnipiano_marl" for run in bundle.runs)


@pytest.mark.parametrize("track", ["standard", "robust", "safety", "marl"])
def test_every_reference_parameter_template_binds_without_ignored_fields(
    track: str,
) -> None:
    base = json.loads((EXAMPLES / f"{track}.json").read_text(encoding="utf-8"))
    for capability in get_track_capability(track).algorithms:
        if not capability.selectable:
            continue
        if track == "marl" and capability.name == "ppo-monolithic":
            continue  # Requires the dedicated one-agent custom task tested below.
        document = json.loads(json.dumps(base))
        document["algorithm"] = {
            "name": capability.name,
            "parameters": capability.parameter_template,
        }
        bundle = bind_plan(parse_request(document))
        assert all(run.launch_status == "bound" for run in bundle.runs)


def test_standard_command_round_trips_through_backend_parser() -> None:
    pytest.importorskip("sb3_contrib")
    from examples.run_sb3_baseline import _build_arg_parser

    run = bind_plan(load_request(EXAMPLES / "standard.json")).runs[0]
    args = _build_arg_parser().parse_args(run.command[2:])
    assert args.algo == "ppo"
    assert args.seed == 0
    assert args.total_steps == 10_000_000
    assert args.gamma == 0.8


def test_robust_command_round_trips_through_backend_parser() -> None:
    pytest.importorskip("torchrl")
    from examples.run_torchrl_robust import _build_arg_parser

    run = bind_plan(load_request(EXAMPLES / "robust.json")).runs[0]
    args = _build_arg_parser().parse_args(run.command[2:])
    assert args.algo == "sac"
    assert args.seed == 0
    assert args.total_steps == 10_000_000
    assert args.gamma == 0.8


@pytest.mark.parametrize(
    ("track", "script", "command_offset"),
    [
        ("standard", ROOT / "examples" / "run_sb3_baseline.py", 2),
        ("robust", ROOT / "examples" / "run_torchrl_robust.py", 2),
    ],
)
def test_script_adapter_flags_exist_in_backend_parser(
    track: str, script: Path, command_offset: int
) -> None:
    run = bind_plan(load_request(EXAMPLES / f"{track}.json")).runs[0]
    declared = _declared_options(script)
    emitted = {
        token
        for token in run.command[command_offset:]
        if token.startswith("--")
    }
    assert emitted <= declared


@pytest.mark.parametrize(
    ("track", "script", "command_offset"),
    [
        ("safety", ROOT / "omnipiano" / "safety" / "run.py", 3),
        (
            "marl",
            ROOT / "omnipiano" / "multiagent" / "training" / "config.py",
            3,
        ),
    ],
)
def test_module_adapter_flags_exist_in_backend_parser(
    track: str, script: Path, command_offset: int
) -> None:
    run = bind_plan(load_request(EXAMPLES / f"{track}.json")).runs[0]
    declared = _declared_options(script)
    emitted = {
        ("--" + token.removeprefix("--no-") if token.startswith("--no-") else token)
        for token in run.command[command_offset:]
        if token.startswith("--")
    }
    assert emitted <= declared


def test_marl_registered_command_round_trips_through_backend_parser() -> None:
    from omnipiano.multiagent.training.config import _parse_args, _resolve_args

    run = bind_plan(load_request(EXAMPLES / "marl.json")).runs[0]
    args = _parse_args(run.command[3:])
    spec = _resolve_args(args)
    assert spec.name == "ippo"
    assert args.seed == 0
    assert args.total_steps == 10_000_000
    assert args.gamma == 0.8


def _custom_marl_document() -> dict:
    document = json.loads((EXAMPLES / "marl.json").read_text(encoding="utf-8"))
    document["task"] = {
        "source": "custom",
        "music": {"source": "repertoire", "piece": "WinterWind"},
        "embodiment": {
            "num_hands": 4,
            "layout_preset": "FourHand",
            "mobility": "custom",
            "hands": [
                {"id": 0, "name": "lh_b", "side": "left", "mobile": True,
                 "initial_key_range": [1, 22]},
                {"id": 1, "name": "rh_b", "side": "right", "mobile": True,
                 "initial_key_range": [23, 44]},
                {"id": 2, "name": "lh_t", "side": "left", "mobile": True,
                 "initial_key_range": [45, 66]},
                {"id": 3, "name": "rh_t", "side": "right", "mobile": True,
                 "initial_key_range": [67, 88]},
            ],
        },
    }
    document["design"]["config"] = {
        "num_agents": 2,
        "assignment": "explicit",
        "sustain_owner": "agent_1",
        "agents": [
            {
                "name": "agent_1", "hand_ids": [0, 1],
                "action_key_range": [1, 44],
                "observation_key_range": [1, 49],
                "visible_teammate_hands": [2],
            },
            {
                "name": "agent_2", "hand_ids": [2, 3],
                "action_key_range": [45, 88],
                "observation_key_range": [40, 88],
                "visible_teammate_hands": [1],
            },
        ],
        "coordination": {"inter_agent_collision_penalty_coefficient": 0.1},
    }
    document["algorithm"]["parameters"] = {
        "learning_rate": 3e-4,
        "hidden_sizes": [256, 256],
    }
    return document


def test_custom_marl_artifact_is_accepted_by_existing_compiler(tmp_path: Path) -> None:
    from omnipiano.multiagent.algos import list_algos
    from omnipiano.multiagent.compile import compile_experiment

    bundle = bind_plan(parse_request(_custom_marl_document()))
    run = bundle.runs[0]
    assert len(run.artifacts) == 1
    artifact = next(iter(run.artifacts.values()))
    assert artifact["schema_version"] == 2
    assert artifact["task"]["agents"][0]["name"] == "agent_1"

    path = tmp_path / "backend_config.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    compiled = compile_experiment(path, registered_algorithms=list_algos())
    assert compiled.algorithm == "ippo"
    assert [agent.name for agent in compiled.task.agents] == ["agent_1", "agent_2"]
    assert compiled.values["inter_agent_collision_penalty_coef"] == 0.1

    from omnipiano.multiagent.training.config import _parse_args, _resolve_args

    command = list(run.command)
    command[3] = str(path)
    args = _parse_args(command[3:])
    _resolve_args(args)
    assert args.lr == 3e-4
    assert args.hidden_sizes_parsed == (256, 256)


def test_ppo_monolithic_binds_with_one_agent_custom_task() -> None:
    document = _custom_marl_document()
    document["algorithm"] = {"name": "ppo-monolithic"}
    document["design"]["config"] = {
        "num_agents": 1,
        "assignment": "explicit",
        "sustain_owner": "agent_1",
        "agents": [
            {
                "name": "agent_1",
                "hand_ids": [0, 1, 2, 3],
                "action_key_range": [1, 88],
                "observation_key_range": [1, 88],
                "visible_teammate_hands": [],
            }
        ],
    }
    bundle = bind_plan(parse_request(document))
    artifact = next(iter(bundle.runs[0].artifacts.values()))
    assert artifact["task"]["num_agents"] == 1
    assert artifact["experiment"]["algo"] == "ppo-monolithic"


def test_standard_adapter_routes_complete_constructor_parameters() -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text(encoding="utf-8"))
    document["algorithm"]["parameters"] = {
        "learning_rate": 3e-4,
        "net_arch": [128, 128],
    }
    run = bind_plan(parse_request(document)).runs[0]
    index = run.command.index("--algorithm-parameters-json")
    assert json.loads(run.command[index + 1]) == document["algorithm"]["parameters"]


def test_adapter_rejects_nondefault_preset_instead_of_ignoring_it() -> None:
    document = json.loads((EXAMPLES / "robust.json").read_text(encoding="utf-8"))
    document["algorithm"]["preset"] = "paper_default"
    with pytest.raises(AdapterError, match="does not define algorithm preset"):
        bind_plan(parse_request(document))


def test_adapter_rejects_unrepresentable_compute_request() -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text(encoding="utf-8"))
    document["run"]["compute"] = {
        "device": "auto", "num_workers": 4, "num_envs": 16,
    }
    with pytest.raises(AdapterError, match="num_workers must remain 1"):
        bind_plan(parse_request(document))


def test_facmac_parameters_route_to_native_options_not_ppo_flags() -> None:
    document = json.loads((EXAMPLES / "marl.json").read_text(encoding="utf-8"))
    document["algorithm"] = {
        "name": "facmac", "parameters": {
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "replay_capacity": 100_000,
        }
    }
    run = bind_plan(parse_request(document)).runs[0]
    index = run.command.index("--native-options-json")
    assert json.loads(run.command[index + 1]) == document["algorithm"]["parameters"]
    assert "--critic-lr" not in run.command
    from omnipiano.multiagent.training.config import _parse_args, _resolve_args
    from omnipiano.multiagent.training.native import native_options

    args = _parse_args(run.command[3:])
    assert _resolve_args(args).name == "facmac"
    options = native_options(args)
    assert options["critic_lr"] == 1e-3
    assert options["replay_capacity"] == 100_000


def test_p3o_complete_nested_parameters_bind_to_safety_runner() -> None:
    document = json.loads((EXAMPLES / "safety.json").read_text(encoding="utf-8"))
    document["algorithm"] = {
        "name": "P3O",
        "parameters": {
            "algo_cfgs": {"kappa": 12.0, "update_iters": 8},
            "model_cfgs": {"actor": {"hidden_sizes": [128, 128]}},
        },
    }
    run = bind_plan(parse_request(document)).runs[0]
    index = run.command.index("--algorithm-parameters-json")
    assert json.loads(run.command[index + 1]) == document["algorithm"]["parameters"]


def test_on_policy_safety_rejects_off_policy_buffer_size() -> None:
    document = json.loads((EXAMPLES / "safety.json").read_text(encoding="utf-8"))
    document["algorithm"]["parameters"] = {"buffer_size": 100_000}
    with pytest.raises(AdapterError, match="unsupported fields"):
        bind_plan(parse_request(document))


def test_safety_adapter_rejects_suite_identity_mismatch() -> None:
    document = json.loads((EXAMPLES / "safety.json").read_text(encoding="utf-8"))
    document["design"]["config"]["task_index"] = 8
    with pytest.raises(AdapterError, match="task mismatch"):
        bind_plan(parse_request(document))


def test_marl_adapter_rejects_unrepresentable_hand_layout() -> None:
    document = _custom_marl_document()
    document["task"]["embodiment"]["hands"][0]["initial_key_range"] = [1, 23]
    with pytest.raises(AdapterError, match="fixed FourHand backend layout"):
        bind_plan(parse_request(document))
