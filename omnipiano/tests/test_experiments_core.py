"""Regression tests for the backend-neutral experiment control plane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnipiano.experiments import (
    ExperimentValidationError,
    compile_plan,
    load_request,
    parse_request,
    resolve_experiment,
)


EXAMPLES = (
    Path(__file__).resolve().parents[1] / "experiments" / "examples"
)


@pytest.mark.parametrize("track", ["standard", "robust", "safety", "marl"])
def test_all_track_examples_validate_and_resolve(track: str) -> None:
    request = load_request(EXAMPLES / f"{track}.json")
    resolved = resolve_experiment(request).to_dict()

    assert request.design.type == track
    assert resolved["schema_version"] == 3
    assert resolved["algorithm"]["preset"] == "default"
    assert resolved["algorithm"]["parameters"] == {}
    assert resolved["run"]["evaluation"]["frequency"] == 50_000
    assert resolved["run"]["evaluation"]["seed_offset"] == 10_000
    assert resolved["run"]["video"]["enabled"] is False
    assert resolved["run"]["compute"]["device"] == "auto"
    assert resolved["run"]["output"]["root"] == "runs"
    assert resolved["run"]["tracking"]["tags"] == []


def test_plan_fans_out_seeds_without_binding_a_backend() -> None:
    request = load_request(EXAMPLES / "marl.json")
    bundle = compile_plan(request)

    assert bundle.plan_version == 2
    assert [run.seed for run in bundle.runs] == [0, 1, 2]
    assert [run.run_id for run in bundle.runs] == [
        "marl-ippo-example--marl--seed-0",
        "marl-ippo-example--marl--seed-1",
        "marl-ippo-example--marl--seed-2",
    ]
    assert all(run.launch_status == "unbound" for run in bundle.runs)
    assert all(run.backend is None for run in bundle.runs)
    assert all(run.command == () for run in bundle.runs)
    assert all(run.artifacts == {} for run in bundle.runs)
    assert bundle.runs[0].output_dir == (
        "runs/marl-ippo-example/marl/seed-0"
    )


def test_original_request_and_resolved_snapshot_are_isolated() -> None:
    request = load_request(EXAMPLES / "standard.json")
    original = request.to_dict()
    resolved = resolve_experiment(request).to_dict()

    resolved["run"]["total_steps"] = 1
    original["experiment"]["seeds"].append(99)

    assert request.run.total_steps == 10_000_000
    assert request.experiment.seeds == (0, 1, 2)
    assert resolve_experiment(request).to_dict()["run"]["total_steps"] == 10_000_000


def test_unknown_top_level_field_fails_closed() -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text())
    document["backend_command"] = "do not trust me"

    with pytest.raises(ExperimentValidationError, match="Additional properties"):
        parse_request(document)


def test_custom_task_checks_hand_count_and_range_order() -> None:
    document = json.loads((EXAMPLES / "marl.json").read_text())
    document["task"] = {
        "source": "custom",
        "music": {"source": "repertoire", "piece": "WinterWind"},
        "embodiment": {
            "num_hands": 2,
            "layout_preset": "FourHand",
            "hands": [
                {
                    "id": 0,
                    "name": "lh_b",
                    "side": "left",
                    "mobile": True,
                    "initial_key_range": [44, 1],
                }
            ],
        },
    }

    with pytest.raises(ExperimentValidationError) as error:
        parse_request(document)

    message = str(error.value)
    assert "num_hands" in message
    assert "range start must not exceed range end" in message


def test_tracking_none_requires_disabled_mode() -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text())
    document["run"]["tracking"] = {"provider": "none", "mode": "online"}

    with pytest.raises(ExperimentValidationError, match="must be disabled"):
        parse_request(document)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"schema_version": 3, "schema_version": 3}', "duplicate JSON key"),
        ('{"schema_version": NaN}', "non-finite JSON number"),
    ],
)
def test_loader_rejects_ambiguous_nonstandard_json(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ExperimentValidationError, match=message):
        load_request(path)
