"""Safety and API regression tests for the unified launch layer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from omnipiano.experiments import bind_plan, load_request, parse_request
from omnipiano.experiments.api import ExperimentService
from omnipiano.experiments.execution import (
    ExecutionError,
    launch_run,
    materialize_bundle,
)
from omnipiano.experiments.environments import environment_catalog


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "omnipiano" / "experiments" / "examples"


def test_materialization_is_contained_atomic_and_idempotent(tmp_path: Path) -> None:
    bundle = bind_plan(load_request(EXAMPLES / "standard.json"))
    first = materialize_bundle(
        bundle, workspace=tmp_path, allowed_root=tmp_path / "runs"
    )
    second = materialize_bundle(
        bundle, workspace=tmp_path, allowed_root=tmp_path / "runs"
    )

    assert first.to_dict() == second.to_dict()
    assert len(first.runs) == 3
    for run in first.runs:
        assert run.manifest_path.is_file()
        assert run.output_dir.is_relative_to(tmp_path / "runs")
        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"] == list(run.command)
        assert manifest["output_dir"] == str(run.output_dir)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    materialized = materialize_bundle(
        bind_plan(load_request(EXAMPLES / "standard.json")),
        workspace=tmp_path,
        allowed_root=tmp_path / "runs",
        dry_run=True,
    )
    assert materialized.dry_run is True
    assert not (tmp_path / "runs").exists()


def test_output_path_escape_is_rejected(tmp_path: Path) -> None:
    document = json.loads((EXAMPLES / "standard.json").read_text(encoding="utf-8"))
    document["run"]["output"] = {"root": "../escape"}
    bundle = bind_plan(parse_request(document))
    with pytest.raises(ExecutionError, match="escapes allowed output root"):
        materialize_bundle(bundle, workspace=tmp_path)
    assert not (tmp_path.parent / "escape").exists()


def test_launch_uses_argument_vector_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = materialize_bundle(
        bind_plan(load_request(EXAMPLES / "standard.json")),
        workspace=tmp_path,
        allowed_root=tmp_path / "runs",
    ).runs[0]
    observed: dict = {}

    class FakeProcess:
        pid = 424242

    def fake_popen(command, **kwargs):  # noqa: ANN001
        observed["command"] = command
        observed.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("omnipiano.experiments.execution.subprocess.Popen", fake_popen)
    monkeypatch.setattr("omnipiano.experiments.execution._process_matches", lambda pid, command: True)
    status = launch_run(run)

    assert observed["command"] == list(run.command)
    assert observed["shell"] is False
    assert observed["cwd"] == tmp_path
    assert status["state"] == "running"
    with pytest.raises(ExecutionError, match="already has launcher state"):
        launch_run(run)


def test_local_service_validates_and_dry_runs_all_seeds(tmp_path: Path) -> None:
    service = ExperimentService(workspace=ROOT, allowed_root=tmp_path)
    document = json.loads((EXAMPLES / "standard.json").read_text(encoding="utf-8"))
    document["run"]["output"] = {"root": str(tmp_path)}

    validated = service.validate(document)
    preview = service.launch(document, dry_run=True)
    assert validated["ok"] is True
    assert validated["run_count"] == 3
    assert preview["dry_run"] is True
    assert len(preview["runs"]) == 3
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("track", ("standard", "robust", "safety", "marl"))
def test_environment_catalog_is_unique_and_dependency_light(track: str) -> None:
    catalog = environment_catalog(track)
    ids = [item["id"] for item in catalog["environments"]]
    assert catalog["count"] == len(ids)
    assert ids
    assert len(ids) == len(set(ids))
    assert "omnipiano.envs.registration" not in sys.modules


@pytest.mark.parametrize("track", ("standard", "robust", "safety", "marl"))
def test_local_service_can_bind_every_canonical_example(
    track: str, tmp_path: Path
) -> None:
    service = ExperimentService(workspace=ROOT, allowed_root=tmp_path)
    document = json.loads((EXAMPLES / f"{track}.json").read_text(encoding="utf-8"))
    document["run"]["output"] = {"root": str(tmp_path)}
    result = service.validate(document)
    assert result["ok"] is True
    assert result["track"] == track
