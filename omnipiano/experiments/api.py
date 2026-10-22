"""Loopback-only JSON API for the unified experiment composer."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from omnipiano.experiments.adapters import AdapterError
from omnipiano.experiments.catalog import CapabilityError, catalog_dict
from omnipiano.experiments.compiler import bind_document
from omnipiano.experiments.environments import (
    EnvironmentCatalogError,
    environment_catalog,
)
from omnipiano.experiments.execution import (
    ExecutionError,
    discover_runs,
    launch_bundle,
    materialize_bundle,
    stop_process,
)
from omnipiano.experiments.schema import ExperimentValidationError


MAX_REQUEST_BYTES = 2 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ApiRequestError(ValueError):
    """A client request is malformed or addresses an unknown resource."""


class ExperimentService:
    """Pure-ish service boundary used by both HTTP handlers and tests."""

    def __init__(self, *, workspace: str | Path, allowed_root: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        root = Path(allowed_root).expanduser()
        self.allowed_root = (root if root.is_absolute() else self.workspace / root).resolve(
            strict=False
        )

    def validate(self, document: Mapping[str, Any]) -> dict[str, Any]:
        # Binding is the readiness gate: a schema-valid request that the
        # current backend cannot represent must not appear launchable in UI.
        bundle = bind_document(dict(document))
        return {
            "ok": True,
            "track": bundle.resolved_experiment.track,
            "run_count": len(bundle.runs),
            "resolved_experiment": bundle.resolved_experiment.to_dict(),
        }

    def plan(self, document: Mapping[str, Any]) -> dict[str, Any]:
        return bind_document(dict(document)).to_dict()

    def materialize(
        self, document: Mapping[str, Any], *, dry_run: bool = False
    ) -> dict[str, Any]:
        bundle = bind_document(dict(document))
        return materialize_bundle(
            bundle,
            workspace=self.workspace,
            allowed_root=self.allowed_root,
            dry_run=dry_run,
        ).to_dict()

    def launch(
        self, document: Mapping[str, Any], *, dry_run: bool = False
    ) -> dict[str, Any]:
        bundle = bind_document(dict(document))
        materialized = materialize_bundle(
            bundle,
            workspace=self.workspace,
            allowed_root=self.allowed_root,
            dry_run=dry_run,
        )
        return launch_bundle(materialized, dry_run=dry_run)

    def runs(self) -> list[dict[str, Any]]:
        return discover_runs(self.allowed_root)

    def stop(self, run_id: str) -> dict[str, Any]:
        matches = [item for item in self.runs() if item.get("run_id") == run_id]
        if len(matches) != 1:
            raise ApiRequestError(
                f"expected exactly one prepared run named {run_id!r}; found {len(matches)}"
            )
        process_path = matches[0]["process"]["process_path"]
        if not Path(process_path).exists():
            raise ApiRequestError(f"run {run_id!r} has not been launched")
        return stop_process(process_path)


def _error_payload(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": str(exc)}
    if isinstance(exc, ExperimentValidationError):
        payload["issues"] = [
            {"path": issue.path, "message": issue.message} for issue in exc.issues
        ]
    return payload


def make_handler(service: ExperimentService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OmniPianoLocalAPI/1"

        def _json(self, status: HTTPStatus, value: Any) -> None:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _origin_is_local(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return urlparse(origin).hostname in LOOPBACK_HOSTS

        def _body(self) -> dict[str, Any]:
            if not self._origin_is_local():
                raise ApiRequestError("non-loopback browser origins are not allowed")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type != "application/json":
                raise ApiRequestError("Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ApiRequestError("invalid Content-Length") from exc
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ApiRequestError(
                    f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes"
                )
            try:
                value = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ApiRequestError("request body is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ApiRequestError("request body must be a JSON object")
            return value

        def _document(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
            value = body.get("config")
            if not isinstance(value, dict):
                raise ApiRequestError("request requires a config object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/health":
                    self._json(HTTPStatus.OK, {"ok": True, "service": "omnipiano-experiments"})
                    return
                if parsed.path == "/api/catalog":
                    query = parse_qs(parsed.query)
                    track = query.get("track", [None])[0]
                    include = query.get("include_unavailable", ["false"])[0].lower() == "true"
                    self._json(
                        HTTPStatus.OK,
                        catalog_dict(track=track, include_unavailable=include),
                    )
                    return
                if parsed.path == "/api/environments":
                    track = parse_qs(parsed.query).get("track", [None])[0]
                    if track is None:
                        raise ApiRequestError("track query parameter is required")
                    self._json(HTTPStatus.OK, environment_catalog(track))
                    return
                if parsed.path == "/api/runs":
                    self._json(HTTPStatus.OK, {"ok": True, "runs": service.runs()})
                    return
                raise ApiRequestError(f"unknown endpoint: {parsed.path}")
            except ApiRequestError as exc:
                self._json(HTTPStatus.NOT_FOUND, _error_payload(exc))
            except (CapabilityError, EnvironmentCatalogError, ExecutionError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, _error_payload(exc))
            except Exception as exc:  # pragma: no cover - final HTTP boundary
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(exc))

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/validate":
                    result = service.validate(self._document(body))
                elif self.path == "/api/plan":
                    result = {"ok": True, "plan": service.plan(self._document(body))}
                elif self.path == "/api/materialize":
                    result = {
                        "ok": True,
                        "materialized": service.materialize(
                            self._document(body), dry_run=bool(body.get("dry_run", False))
                        ),
                    }
                elif self.path == "/api/launch":
                    result = {
                        "ok": True,
                        "launch": service.launch(
                            self._document(body), dry_run=bool(body.get("dry_run", False))
                        ),
                    }
                elif self.path == "/api/stop":
                    run_id = body.get("run_id")
                    if not isinstance(run_id, str) or not run_id:
                        raise ApiRequestError("request requires a non-empty run_id")
                    result = {"ok": True, "process": service.stop(run_id)}
                else:
                    raise ApiRequestError(f"unknown endpoint: {self.path}")
                self._json(HTTPStatus.OK, result)
            except ApiRequestError as exc:
                self._json(HTTPStatus.BAD_REQUEST, _error_payload(exc))
            except (ExperimentValidationError, CapabilityError, AdapterError, ExecutionError) as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, _error_payload(exc))
            except Exception as exc:  # pragma: no cover - final HTTP boundary
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(exc))

        def log_message(self, format: str, *args: Any) -> None:
            # Keep the local server quiet except for errors surfaced as JSON.
            return

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: str | Path | None = None,
    allowed_root: str | Path = "runs",
) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ApiRequestError(
            "the experiment API is local-only; bind to 127.0.0.1, localhost, or ::1"
        )
    workspace_path = Path(workspace or Path(__file__).resolve().parents[2]).resolve()
    service = ExperimentService(workspace=workspace_path, allowed_root=allowed_root)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"OmniPiano experiment API: http://{host}:{server.server_port}")
    print(f"Allowed output root: {service.allowed_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["ApiRequestError", "ExperimentService", "make_handler", "serve"]
