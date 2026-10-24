"""Capability-catalog drift and compatibility tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from omnipiano.experiments import (
    CapabilityError,
    catalog_dict,
    load_request,
    resolve_experiment,
)
from omnipiano.experiments.catalog import get_track_capability


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "omnipiano" / "experiments" / "examples"


def _dict_assignment_keys(path: Path, variable: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Dict) and all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in value.keys
        ):
            matches.append({key.value for key in value.keys})
    if len(matches) != 1:
        raise AssertionError(f"expected one literal {variable} mapping in {path}")
    return matches[0]


def _catalog_names(track: str, *, include_unavailable: bool = False) -> set[str]:
    payload = catalog_dict(track=track, include_unavailable=include_unavailable)
    return {item["name"] for item in payload["tracks"][track]["algorithms"]}


def test_catalog_is_json_serializable_and_has_four_tracks() -> None:
    payload = catalog_dict()
    assert payload["catalog_version"] == 2
    assert payload["contract_version"] == 3
    assert tuple(payload["tracks"]) == ("standard", "robust", "safety", "marl")
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_standard_catalog_matches_sb3_trainer_registry() -> None:
    backend = _dict_assignment_keys(
        ROOT / "examples" / "run_sb3_baseline.py", "ALGO_REGISTRY"
    )
    assert _catalog_names("standard") == backend


def test_robust_catalog_matches_torchrl_trainer_registry() -> None:
    backend = _dict_assignment_keys(
        ROOT / "examples" / "run_torchrl_robust.py", "ALGO_REGISTRY"
    )
    assert _catalog_names("robust") == backend


def test_safety_catalog_matches_native_omnisafe_registry() -> None:
    from omnipiano.safety.algorithms import SUPPORTED_ALGORITHMS

    assert _catalog_names("safety") == set(SUPPORTED_ALGORITHMS)
    assert len(_catalog_names("safety")) == 32


def test_marl_catalog_matches_algorithm_registry_metadata() -> None:
    from omnipiano.multiagent.algos import get_algo, list_algos

    catalog = get_track_capability("marl")
    assert {item.name for item in catalog.algorithms} == set(list_algos())
    for item in catalog.algorithms:
        backend = get_algo(item.name)
        assert item.status == backend.status
        assert item.backend == backend.backend
        assert item.family == backend.family
        assert item.critic == backend.critic_input
        assert item.execution == backend.execution
        assert item.required_num_agents == backend.required_num_agents


def test_default_catalog_hides_planned_algorithms() -> None:
    assert "mat" not in _catalog_names("marl")
    assert "masac" not in _catalog_names("marl")
    expanded = _catalog_names("marl", include_unavailable=True)
    assert {"mat", "masac"}.issubset(expanded)


def test_plan_rejects_algorithm_track_mismatch() -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text())
    document["algorithm"]["name"] = "mappo"
    from omnipiano.experiments import parse_request

    with pytest.raises(CapabilityError, match="not registered for track"):
        resolve_experiment(parse_request(document))


def test_plan_rejects_planned_algorithm() -> None:
    document = json.loads((EXAMPLES / "marl.json").read_text())
    document["algorithm"]["name"] = "mat"
    from omnipiano.experiments import parse_request

    with pytest.raises(CapabilityError, match="planned"):
        resolve_experiment(parse_request(document))


def test_only_marl_accepts_custom_task_in_current_zero_intrusion_scope() -> None:
    marl = load_request(EXAMPLES / "marl.json")
    standard_document = json.loads((EXAMPLES / "standard.json").read_text())
    standard_document["task"] = {
        "source": "custom",
        "music": {"source": "repertoire", "piece": "WinterWind"},
        "embodiment": {
            "num_hands": 1,
            "layout_preset": "OneHand",
            "hands": [
                {
                    "id": 0,
                    "name": "rh",
                    "side": "right",
                    "mobile": True,
                    "initial_key_range": [1, 88],
                }
            ],
        },
    }
    from omnipiano.experiments import parse_request

    assert "custom" in get_track_capability(marl.design.type).task_sources
    with pytest.raises(CapabilityError, match="does not support task source"):
        resolve_experiment(parse_request(standard_document))


def test_catalog_records_current_track_specific_limits() -> None:
    robust = get_track_capability("robust")
    safety = get_track_capability("safety")
    marl = get_track_capability("marl")

    assert robust.features["noise_channels"] == [
        "action", "observation", "reward", "gravity",
        "contact_friction", "initial_hand_pose",
    ]
    assert set(safety.features["cost_semantics"]) == {
        "joint_range", "actuator_power", "injured_finger", "hand_collision"
    }
    assert marl.features["robust_noise_channels"] == ["observation"]


def test_catalog_parameter_lists_match_adapter_allowlists() -> None:
    from omnipiano.experiments.adapters.marl import (
        NATIVE_PARAMETER_NAMES,
        PARAMETER_FLAGS as MARL_PARAMETERS,
        PPO_PARAMETER_ALGORITHMS,
    )
    from omnipiano.experiments.adapters.robust import ALGORITHM_PARAMETERS as ROBUST
    from omnipiano.experiments.adapters.standard import ALGORITHM_PARAMETERS as STANDARD

    for capability in get_track_capability("standard").algorithms:
        assert set(capability.configurable_parameters) == set(STANDARD[capability.name])
    for capability in get_track_capability("robust").algorithms:
        assert set(capability.configurable_parameters) == set(ROBUST[capability.name])
    for capability in get_track_capability("safety").algorithms:
        from omnipiano.safety.parameters import configurable_sections

        assert set(capability.configurable_parameters) == set(
            configurable_sections(capability.name)
        )
    for capability in get_track_capability("marl").algorithms:
        if capability.status != "supported":
            continue
        expected = set(MARL_PARAMETERS) if capability.name in PPO_PARAMETER_ALGORITHMS else set()
        expected.update(NATIVE_PARAMETER_NAMES.get(capability.name, set()))
        assert set(capability.configurable_parameters) == expected


def test_every_selectable_algorithm_has_a_complete_reference_template() -> None:
    for track in ("standard", "robust", "safety", "marl"):
        for capability in get_track_capability(track).algorithms:
            if not capability.selectable:
                continue
            assert capability.parameter_template
            assert set(capability.parameter_template) <= set(
                capability.configurable_parameters
            )

    p3o = get_track_capability("safety").algorithm("P3O").parameter_template
    assert p3o["algo_cfgs"]["kappa"] == 20.0
    assert p3o["algo_cfgs"]["update_iters"] == 10
    assert p3o["model_cfgs"]["actor"]["hidden_sizes"] == [64, 64]
