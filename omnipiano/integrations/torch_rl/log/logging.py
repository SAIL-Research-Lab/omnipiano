"""Minimal append-only CSV logging."""

from __future__ import annotations

import csv
from pathlib import Path


class ProgressLogger:
    def __init__(
        self,
        path: str | Path,
        min_interval_steps: int = 5_000,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None
        self._min_interval_steps = int(min_interval_steps)
        self._last_written_step = None
        self._pending_row = None
        self._stream = None
        self._writer = None

    def write(self, row: dict, force: bool = False) -> None:
        """Record the latest row, writing at most once per step interval."""
        self._pending_row = dict(row)
        env_steps = int(row["env_steps"])
        if (
            not force
            and self._last_written_step is not None
            and env_steps - self._last_written_step < self._min_interval_steps
        ):
            return

        if self._fields is None:
            self._fields = list(row)
        if self._stream is None:
            self._stream = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._stream, fieldnames=self._fields
            )
            if self._stream.tell() == 0:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._stream.flush()
        self._last_written_step = env_steps
        self._pending_row = None

    def close(self) -> None:
        """Write the final pending row and release the persistent handle."""
        if self._pending_row is not None:
            self.write(self._pending_row, force=True)
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._writer = None
