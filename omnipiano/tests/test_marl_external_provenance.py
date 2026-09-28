"""Verify that vendored reference code has not been edited.

This is the test that converts "we use the unmodified reference implementation"
from a claim into a fact. It is also the test that catches the most likely
accident: an IDE's format-on-save silently reflowing upstream code, after which
an oracle comparison no longer proves what it says it proves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

EXTERNAL = Path(__file__).resolve().parents[1] / "multiagent" / "external"
VENDORED = sorted(
    directory
    for directory in EXTERNAL.iterdir()
    if directory.is_dir()
    and any(
        path.suffix == ".py" and not path.name.startswith("_")
        for path in directory.iterdir()
    )
)


def test_something_is_vendored():
    if not VENDORED:
        pytest.skip(
            "external/ is empty; run "
            "omnipiano/multiagent/external/fetch_reference.sh"
        )


@pytest.mark.parametrize("d", VENDORED, ids=lambda p: p.name)
def test_provenance_is_complete_and_intact(d: Path):
    manifest = d / "PROVENANCE.json"
    assert manifest.is_file(), (
        f"{d.name}/ has vendored files but no PROVENANCE.json; re-run "
        "`bash omnipiano/multiagent/external/fetch_reference.sh "
        f"vendor {d.name}`")
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    for key in ("upstream_url", "upstream_commit", "sha256"):
        assert meta.get(key), f"{d.name}: PROVENANCE.json is missing {key!r}"
    assert len(meta["upstream_commit"]) >= 12, (
        f"{d.name}: upstream_commit must be a full SHA, not a branch name -- a "
        f"branch moves and the provenance would silently stop being true")
    assert any(p.name.startswith("UPSTREAM_LICENSE")
               or p.name.startswith("UPSTREAM_COPYING") for p in d.iterdir()), (
        f"{d.name}: the upstream licence text must be vendored alongside the "
        f"code. MIT and Apache-2.0 both require it; this is a legal obligation.")

    on_disk = {p.name for p in d.iterdir()
               if p.suffix == ".py" and not p.name.startswith("_")}
    recorded = {k for k in meta["sha256"] if k.endswith(".py")
                and not k.startswith("_")}
    assert on_disk == recorded, (
        f"{d.name}: vendored .py files {sorted(on_disk)} do not match "
        f"PROVENANCE {sorted(recorded)}. Files were added or removed by hand.")

    for name, expected in meta["sha256"].items():
        p = d / name
        if not p.is_file():
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{d.name}/{name} HAS BEEN MODIFIED (sha256 {actual[:12]} != "
            f"{expected[:12]}). Vendored reference code is verbatim by policy: "
            f"revert it and put your change in {d.name}/_adapter.py, or the "
            f"oracle tests no longer prove agreement with the reference.")
