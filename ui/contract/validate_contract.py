#!/usr/bin/env python3
"""Validate the isolated UI contract and all checked-in examples.

JSON Schema handles structural validation. The small semantic layer below only
checks relations JSON Schema cannot express cleanly (for example, exact MARL
hand ownership). It intentionally imports no OmniPiano runtime modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover - depends on the caller's env.
    raise SystemExit(
        "Contract validation requires the 'jsonschema' package. "
        "Use the project's MARL environment or install jsonschema."
    ) from exc


CONTRACT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = CONTRACT_DIR / "experiment.schema.json"
SUPPORT_PATH = CONTRACT_DIR / "support.json"
EXAMPLES_DIR = CONTRACT_DIR / "examples"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _check_range(value: list[int], label: str) -> None:
    if value[0] > value[1]:
        raise ValueError(f"{label}: range start must not exceed range end")


def _validate_shared(document: Mapping[str, Any]) -> None:
    embodiment = document["task"]["embodiment"]
    hands = embodiment["hands"]
    hand_ids = [hand["id"] for hand in hands]

    if len(hands) != embodiment["num_hands"]:
        raise ValueError("task.embodiment.num_hands must equal len(hands)")
    if len(hand_ids) != len(set(hand_ids)):
        raise ValueError("task.embodiment.hands must use unique hand IDs")

    for hand in hands:
        if hand["initial_key_range"] is not None:
            _check_range(
                hand["initial_key_range"],
                f"task.embodiment.hands[{hand['id']}].initial_key_range",
            )

    evaluation = document["evaluation"]
    video = evaluation["video"]
    if video["enabled"] and video["frequency"] == 0:
        raise ValueError("evaluation.video.frequency must be positive when enabled")

    tracking = document["tracking"]
    if tracking["provider"] == "none" and tracking["mode"] != "disabled":
        raise ValueError("tracking.mode must be disabled when provider is none")
    if tracking["provider"] == "wandb" and tracking["mode"] == "disabled":
        raise ValueError("tracking.mode must be offline or online for wandb")


def _validate_robust(config: Mapping[str, Any]) -> None:
    channels = config["channels"]
    targets = [channel["target"] for channel in channels]
    if len(targets) != len(set(targets)):
        raise ValueError("robust channels must have unique targets")

    for channel in channels:
        magnitude = channel["magnitude"]
        distribution = channel["distribution"]
        label = f"robust channel {channel['target']}"
        if channel["target"] == "initial_hand_pose":
            if not {"y_mm", "z_mm"}.issubset(magnitude):
                raise ValueError(
                    f"{label}: initial_hand_pose requires magnitude.y_mm and .z_mm"
                )
        elif distribution in {"gaussian", "constant_shift"}:
            if "value" not in magnitude:
                raise ValueError(f"{label}: {distribution} requires magnitude.value")
        elif not {"low", "high"}.issubset(magnitude):
            raise ValueError(
                f"{label}: bounded_uniform requires magnitude.low and .high"
            )
        elif magnitude["low"] > magnitude["high"]:
            raise ValueError(f"{label}: magnitude.low must not exceed .high")


def _validate_safety(
    document: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    cost = config["primary_cost"]
    if cost["type"] == "power_spike":
        raise ValueError("power_spike has no finalized cost definition")
    hand_ids = {hand["id"] for hand in document["task"]["embodiment"]["hands"]}
    if not set(cost["target_hand_ids"]).issubset(hand_ids):
        raise ValueError("safety target_hand_ids must reference configured hands")

    if cost["type"].startswith("range_"):
        if not cost["target_joints"]:
            raise ValueError("range costs require at least one target joint")
        if cost["safe_interval_fraction"] is None:
            raise ValueError("range costs require safe_interval_fraction")
    else:
        if cost["power_threshold"] is None:
            raise ValueError("power costs require power_threshold")
        energy = document["task"]["reward"]["energy_regularization"]
        if energy["enabled"]:
            raise ValueError(
                "power-cost safety tasks must disable reward energy regularization"
            )


def _validate_marl(
    document: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    agents = config["agents"]
    expected_critic = {
        "ippo": "local",
        "ippo-rllib-module": "local",
        "mappo": "centralized",
        "mappo-own-critic": "own_observation_ablation",
        "mappo-noshuffle": "centralized",
        "happo": "centralized",
        "happo-m1-control": "centralized",
    }.get(document["algorithm"]["name"])
    if expected_critic is not None and config["critic_mode"] != expected_critic:
        raise ValueError("marl critic_mode must match the chosen algorithm")
    if len(agents) != config["num_agents"]:
        raise ValueError("marl num_agents must equal len(agents)")

    names = [agent["name"] for agent in agents]
    if len(names) != len(set(names)):
        raise ValueError("marl agent names must be unique")
    if config["sustain_owner"] not in names:
        raise ValueError("marl sustain_owner must name a configured agent")

    expected_hands = {
        hand["id"] for hand in document["task"]["embodiment"]["hands"]
    }
    owned_hands = [hand_id for agent in agents for hand_id in agent["hand_ids"]]
    if len(owned_hands) != len(set(owned_hands)):
        raise ValueError("marl hand ownership must not overlap")
    if set(owned_hands) != expected_hands:
        raise ValueError("marl agents must own every configured hand exactly once")

    for agent in agents:
        _check_range(
            agent["action_key_range"],
            f"marl agent {agent['name']} action_key_range",
        )
        _check_range(
            agent["observation_key_range"],
            f"marl agent {agent['name']} observation_key_range",
        )
        visible = agent["visible_teammate_hands"]
        if isinstance(visible, list):
            if not set(visible).issubset(expected_hands):
                raise ValueError(
                    f"marl agent {agent['name']} references an unknown visible hand"
                )
            if set(visible).intersection(agent["hand_ids"]):
                raise ValueError(
                    f"marl agent {agent['name']} lists an owned hand as a teammate"
                )


def validate_semantics(document: Mapping[str, Any]) -> None:
    """Validate cross-field invariants shared by the future UI and exporters."""
    _validate_shared(document)
    design = document["design"]
    if design["type"] == "robust":
        _validate_robust(design["config"])
    elif design["type"] == "safety":
        _validate_safety(document, design["config"])
    elif design["type"] == "marl":
        _validate_marl(document, design["config"])


def main() -> None:
    schema = _read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    support = _read_json(SUPPORT_PATH)
    if support["contract_version"] != schema["properties"]["schema_version"]["const"]:
        raise ValueError("support.json and experiment schema versions differ")

    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))
    expected_tracks = {"standard", "robust", "safety", "marl"}
    actual_tracks: set[str] = set()

    for path in example_paths:
        document = _read_json(path)
        errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
        if errors:
            details = "\n".join(
                f"  {'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
                for error in errors
            )
            raise ValueError(f"{path.name} failed JSON Schema validation:\n{details}")

        validate_semantics(document)

        # A stable load/dump/load cycle is the minimum import/export guarantee.
        if json.loads(json.dumps(document, sort_keys=True)) != document:
            raise ValueError(f"{path.name} failed JSON round-trip validation")

        track = document["design"]["type"]
        if path.stem != track:
            raise ValueError(f"{path.name} contains design.type={track!r}")
        actual_tracks.add(track)
        print(f"validated {path.relative_to(CONTRACT_DIR)}")

    if actual_tracks != expected_tracks:
        raise ValueError(
            f"examples must cover {sorted(expected_tracks)}, got {sorted(actual_tracks)}"
        )

    print("contract schema and all four track examples are valid")


if __name__ == "__main__":
    main()
