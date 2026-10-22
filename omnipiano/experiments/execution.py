"""Safe materialization and process control for bound experiment plans.

This module is deliberately separate from every training backend.  It only
accepts :class:`PlanBundle` objects produced by the in-repository adapters,
confines all writes to an explicit output root, and launches argument vectors
with ``shell=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from omnipiano.experiments.models import PlanBundle, RunPlan


CONTROL_DIR = ".omnipiano"
MANIFEST_NAME = "launch.json"
PROCESS_NAME = "process.json"
STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"
MAX_STOP_WAIT_SECONDS = 10.0


class ExecutionError(RuntimeError):
    """A bound plan cannot be safely materialized, launched, or stopped."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes, *, idempotent: bool = False) -> None:
    """Create ``path`` atomically without ever replacing different content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if idempotent and path.is_file() and path.read_bytes() == payload:
            return
        raise ExecutionError(f"refusing to overwrite existing file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            if idempotent and path.is_file() and path.read_bytes() == payload:
                return
            raise ExecutionError(f"refusing to overwrite existing file: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _replace_json(path: Path, value: Any) -> None:
    """Atomically replace launcher-owned state (never a backend artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _contained(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = root.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ExecutionError(f"{label} escapes allowed output root {root}: {resolved}") from exc
    return resolved


def _configured_output_root(bundle: PlanBundle, workspace: Path) -> Path:
    configured = Path(bundle.resolved_experiment.to_dict()["run"]["output"]["root"])
    return configured if configured.is_absolute() else workspace / configured


def _validate_command(run: RunPlan) -> None:
    expected: dict[str, tuple[str, ...]] = {
        "stable_baselines3": ("python", "examples/run_sb3_baseline.py"),
        "torchrl": ("python", "examples/run_torchrl_robust.py"),
        "omnisafe": ("python", "-m", "omnipiano.safety.run"),
        "omnipiano_marl": ("python", "-m", "omnipiano.multiagent.train"),
    }
    if run.launch_status != "bound" or run.backend not in expected:
        raise ExecutionError(f"run {run.run_id!r} is not bound to a supported backend")
    prefix = expected[run.backend]
    if run.command[: len(prefix)] != prefix:
        raise ExecutionError(
            f"run {run.run_id!r} has an unexpected command for backend {run.backend}"
        )
    if not run.command or any(not isinstance(token, str) or "\x00" in token for token in run.command):
        raise ExecutionError(f"run {run.run_id!r} contains an invalid command token")


@dataclass(frozen=True)
class MaterializedRun:
    run_id: str
    output_dir: Path
    control_dir: Path
    manifest_path: Path
    command: tuple[str, ...]
    environment: Mapping[str, str]
    backend: str
    seed: int
    workspace: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "backend": self.backend,
            "seed": self.seed,
            "workspace": str(self.workspace),
            "command": list(self.command),
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class MaterializedBundle:
    workspace: Path
    allowed_root: Path
    dry_run: bool
    runs: tuple[MaterializedRun, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "workspace": str(self.workspace),
            "allowed_root": str(self.allowed_root),
            "run_count": len(self.runs),
            "runs": [run.to_dict() for run in self.runs],
        }


def materialize_bundle(
    bundle: PlanBundle,
    *,
    workspace: str | Path | None = None,
    allowed_root: str | Path | None = None,
    dry_run: bool = False,
) -> MaterializedBundle:
    """Resolve paths and optionally write adapter artifacts and launch manifests."""
    workspace_path = Path(workspace or _repo_root()).expanduser().resolve()
    configured_root = _configured_output_root(bundle, workspace_path).resolve(strict=False)
    if allowed_root is None:
        # Default policy: configured output must remain under the repository.
        _contained(configured_root, workspace_path, label="configured output root")
        allowed_path = configured_root
    else:
        allowed_candidate = Path(allowed_root)
        if not allowed_candidate.is_absolute():
            allowed_candidate = workspace_path / allowed_candidate
        allowed_path = allowed_candidate.expanduser().resolve(strict=False)
        _contained(configured_root, allowed_path, label="configured output root")

    materialized: list[MaterializedRun] = []
    seen_outputs: set[Path] = set()
    for run in bundle.runs:
        _validate_command(run)
        raw_output = Path(run.output_dir)
        output = raw_output if raw_output.is_absolute() else workspace_path / raw_output
        output = _contained(output, allowed_path, label=f"run {run.run_id} output_dir")
        if output in seen_outputs:
            raise ExecutionError(f"multiple runs resolve to the same output directory: {output}")
        seen_outputs.add(output)

        replacements = {run.output_dir: str(output)}
        resolved_artifacts: list[tuple[Path, Any]] = []
        for raw_path, value in run.artifacts.items():
            artifact = Path(raw_path)
            artifact = artifact if artifact.is_absolute() else workspace_path / artifact
            artifact = _contained(artifact, output, label=f"run {run.run_id} artifact")
            replacements[str(raw_path)] = str(artifact)
            resolved_artifacts.append((artifact, value))

        command = tuple(replacements.get(token, token) for token in run.command)
        environment = {str(key): str(value) for key, value in run.environment.items()}
        control = output / CONTROL_DIR
        manifest = control / MANIFEST_NAME
        item = MaterializedRun(
            run_id=run.run_id,
            output_dir=output,
            control_dir=control,
            manifest_path=manifest,
            command=command,
            environment=environment,
            backend=str(run.backend),
            seed=run.seed,
            workspace=workspace_path,
        )
        materialized.append(item)

        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            for path, value in resolved_artifacts:
                _atomic_write(path, _json_bytes(value), idempotent=True)
            _atomic_write(
                manifest,
                _json_bytes(
                    {
                        "manifest_version": 1,
                        **item.to_dict(),
                    }
                ),
                idempotent=True,
            )

    return MaterializedBundle(
        workspace=workspace_path,
        allowed_root=allowed_path,
        dry_run=dry_run,
        runs=tuple(materialized),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot read launcher state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"launcher state is not a JSON object: {path}")
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_matches(pid: int, command: Sequence[str]) -> bool:
    """Defend stop/duplicate checks against PID reuse."""
    if not _pid_alive(pid):
        return False
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        try:
            actual = tuple(
                token.decode("utf-8", errors="replace")
                for token in proc_cmdline.read_bytes().split(b"\0")
                if token
            )
            return actual == tuple(command)
        except OSError:
            return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    rendered = result.stdout.strip()
    return result.returncode == 0 and all(token in rendered for token in command[:3])


def _process_path(run: MaterializedRun | Path) -> Path:
    if isinstance(run, MaterializedRun):
        return run.control_dir / PROCESS_NAME
    path = Path(run)
    return path if path.name == PROCESS_NAME else path / CONTROL_DIR / PROCESS_NAME


def process_status(run: MaterializedRun | Path) -> dict[str, Any]:
    path = _process_path(run)
    if not path.exists():
        return {"state": "prepared", "process_path": str(path)}
    record = _read_json(path)
    if record.get("state") == "running":
        pid = int(record.get("pid", -1))
        command = tuple(record.get("command", []))
        if not _process_matches(pid, command):
            record = {**record, "state": "exited", "observed_at": _utc_now()}
            _replace_json(path, record)
    return {**record, "process_path": str(path)}


def launch_run(run: MaterializedRun) -> dict[str, Any]:
    """Start one prepared run as a detached process."""
    if not run.manifest_path.exists():
        raise ExecutionError(f"run is not materialized: {run.run_id}")
    process_path = _process_path(run)
    if process_path.exists():
        status = process_status(run)
        raise ExecutionError(
            f"run {run.run_id!r} already has launcher state {status['state']!r}; "
            "use a fresh experiment name/output directory"
        )

    run.control_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run.control_dir / STDOUT_NAME
    stderr_path = run.control_dir / STDERR_NAME
    environment = os.environ.copy()
    environment.update(run.environment)
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                list(run.command),
                cwd=run.workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise ExecutionError(f"failed to launch {run.run_id!r}: {exc}") from exc
    record = {
        "state": "running",
        "run_id": run.run_id,
        "pid": process.pid,
        "started_at": _utc_now(),
        "command": list(run.command),
        "cwd": str(run.workspace),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    _atomic_write(process_path, _json_bytes(record))
    time.sleep(0.05)
    return process_status(run)


def launch_bundle(bundle: MaterializedBundle, *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run or bundle.dry_run:
        return {"dry_run": True, "runs": [run.to_dict() for run in bundle.runs]}
    # Preflight every run before starting any process.
    for run in bundle.runs:
        if not run.manifest_path.exists():
            raise ExecutionError(f"run is not materialized: {run.run_id}")
        if _process_path(run).exists():
            status = process_status(run)
            raise ExecutionError(
                f"run {run.run_id!r} already has launcher state {status['state']!r}"
            )
    return {"dry_run": False, "runs": [launch_run(run) for run in bundle.runs]}


def stop_process(process_path: str | Path, *, wait_seconds: float = MAX_STOP_WAIT_SECONDS) -> dict[str, Any]:
    path = Path(process_path).expanduser().resolve()
    record = _read_json(path)
    if record.get("state") != "running":
        return {**record, "process_path": str(path)}
    pid = int(record.get("pid", -1))
    command = tuple(record.get("command", []))
    if not _process_matches(pid, command):
        stale = {**record, "state": "exited", "observed_at": _utc_now()}
        _replace_json(path, stale)
        return {**stale, "process_path": str(path)}

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    state = "stopping" if _pid_alive(pid) else "stopped"
    updated = {**record, "state": state, "stop_requested_at": _utc_now()}
    _replace_json(path, updated)
    return {**updated, "process_path": str(path)}


def discover_runs(allowed_root: str | Path) -> list[dict[str, Any]]:
    root = Path(allowed_root).expanduser().resolve(strict=False)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for manifest in sorted(root.rglob(f"{CONTROL_DIR}/{MANIFEST_NAME}")):
        _contained(manifest, root, label="manifest")
        value = _read_json(manifest)
        status = process_status(manifest.parent / PROCESS_NAME)
        records.append(
            {
                "run_id": value.get("run_id"),
                "backend": value.get("backend"),
                "seed": value.get("seed"),
                "output_dir": value.get("output_dir"),
                "manifest_path": str(manifest),
                "process": status,
            }
        )
    return records
