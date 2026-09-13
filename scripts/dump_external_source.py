#!/usr/bin/env python
"""Print the source of a vendored reference symbol, for writing an adapter.

    python scripts/dump_external_source.py harl happo
    python scripts/dump_external_source.py on_policy r_mappo

Reads the VENDORED copy, not the upstream clone, so what you see is exactly what
provenance_test.py is hashing.
"""

from __future__ import annotations

import pathlib
import re
import sys

VENDOR = pathlib.Path("omnipiano/multiagent/external")


def main(name: str, needle: str) -> int:
    d = VENDOR / name
    if not d.is_dir():
        print(f"not vendored: {d}\n  bash scripts/fetch_external.sh vendor {name}")
        return 1
    hits = [p for p in sorted(d.glob("*.py")) if needle.lower() in p.name.lower()]
    if not hits:
        hits = [p for p in sorted(d.glob("*.py"))
                if needle.lower() in p.read_text(encoding="utf-8", errors="replace").lower()]
    if not hits:
        print(f"no vendored file in {d} matches {needle!r}")
        return 1
    for p in hits:
        text = p.read_text(encoding="utf-8", errors="replace")
        print(f"\n{'=' * 78}\n=== {p}  ({len(text.splitlines())} lines)\n{'=' * 78}")
        print(text)
        # The three things an adapter must pin down; surface them explicitly.
        for label, pat in (("ratio/exp", r"\bexp\b|ratio"),
                           ("clamp/clip", r"clamp|clip"),
                           ("factor", r"factor|imp_weights|prod"),
                           ("reduction", r"\.mean\(|\.sum\(")):
            found = sorted({m.group(0) for m in re.finditer(pat, text)})
            print(f"# hint[{label}]: {found}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))