"""Repository-anchored paths.

Run artifacts must land in the same place no matter what the shell's working
directory is -- otherwise ``--run-dir examples/logs/x`` means different things
depending on where you launched from, which is exactly how a CLI silently
scatters an experiment across the filesystem.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory containing ``setup.py`` and the ``omnipiano`` package."""
    return Path(__file__).resolve().parents[2]


def logs_root() -> Path:
    """Canonical run-artifact root: ``<repo>/examples/logs``."""
    return repo_root() / "examples" / "logs"


def resolve_run_path(path: str | Path) -> Path:
    """Resolve a user path, falling back to a repo-relative interpretation.

    ``--run-dir examples/logs/foo`` therefore works from any cwd, while an
    absolute path or an existing relative path is honoured unchanged.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    from_repo = (repo_root() / candidate).resolve()
    if from_repo.exists() or from_repo.parent.exists():
        return from_repo
    return candidate.resolve()