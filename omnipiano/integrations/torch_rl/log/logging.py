"""Minimal append-only CSV logging."""

from __future__ import annotations

import csv
from pathlib import Path


class ProgressLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None

    def write(self, row: dict) -> None:
        if self._fields is None:
            self._fields = list(row)
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fields)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
