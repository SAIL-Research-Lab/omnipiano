"""Dependency-light environment choices for the local experiment UI.

The control plane must not initialize MuJoCo merely to populate a dropdown.
The small generated families below mirror the registrations in
``omnipiano/envs/__init__.py`` and drift tests cover the canonical examples.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any

from omnipiano.experiments.catalog import TRACKS


class EnvironmentCatalogError(ValueError):
    """The environment catalog cannot serve the requested track."""


def _general_ids() -> set[str]:
    # Mirrors _register_general_rl_forelise() and
    # _register_general_rl_song_ladder() in envs/__init__.py.
    result = {
        "OmniPiano-ForElise-OneHand-GeneralRL-v0",
        "OmniPiano-ForElise-TwoHand-GeneralRL-v0",
    }
    result.update(
        f"OmniPiano-ForElise-{hands}Hand-StaticPartition-GeneralRL-v0"
        for hands in ("Three", "Four", "Five")
    )
    for piece in ("TwinkleTwinkleLittleStar", "WinterWind", "PicturesGreatKiev"):
        result.update(
            {
                f"OmniPiano-{piece}-OneHand-GeneralRL-v0",
                f"OmniPiano-{piece}-TwoHand-GeneralRL-v0",
            }
        )
        for hands in ("Three", "Four", "Five"):
            result.add(f"OmniPiano-{piece}-{hands}HandPrototype-GeneralRL-v0")
            result.add(
                f"OmniPiano-{piece}-{hands}Hand-StaticPartition-GeneralRL-v0"
            )
    return result


def _robust_ids() -> set[str]:
    result = {"OmniPiano-ClairDeLune-Clean-v0"}
    levels = {"A": (5, 10, 15), "O": (10, 20, 30), "R": (10, 30, 50)}
    for letter, values in levels.items():
        for distribution in ("Gauss", "Uniform", "Shift"):
            for level in values:
                result.add(
                    f"OmniPiano-ClairDeLune-{letter}-{distribution}-P{level:02d}-v0"
                )
    result.update(
        {
            "OmniPiano-ClairDeLune-A-Shift-N15-v0",
            "OmniPiano-ClairDeLune-O-Shift-N15-v0",
            "OmniPiano-ClairDeLune-R-Shift-N50-v0",
            "OmniPiano-ClairDeLune-R-Shift-P50-NoOAR-v0",
            "OmniPiano-ClairDeLune-R-Shift-N50-NoOAR-v0",
            "OmniPiano-ClairDeLune-AO-Gauss-P15-v0",
            "OmniPiano-ClairDeLune-AR-Uniform-A15-R50-v0",
            "OmniPiano-ClairDeLune-OR-Shift-ON15-RN50-v0",
            "OmniPiano-ClairDeLune-Clean-NoOAR-v0",
            "OmniPiano-FantaisieImpromptu-ActionRobust-v0",
            "OmniPiano-ClairDeLune-ObservationRobust-v0",
        }
    )
    for distribution in ("Gauss", "Uniform", "Shift"):
        for level in (200, 300, 400):
            result.add(f"OmniPiano-ClairDeLune-G-{distribution}-P{level:02d}-v0")
            if distribution != "Shift":
                result.add(
                    f"OmniPiano-ClairDeLune-G-Step-{distribution}-P{level:02d}-v0"
                )
        for level in (20, 30, 40):
            result.add(f"OmniPiano-ClairDeLune-CF-{distribution}-P{level:02d}-v0")
            if distribution != "Shift":
                result.add(
                    f"OmniPiano-ClairDeLune-CF-Step-{distribution}-P{level:02d}-v0"
                )
        for y_level, z_level in ((10, 5), (25, 10), (50, 20)):
            result.add(
                f"OmniPiano-ClairDeLune-HP-{distribution}-"
                f"Y{y_level:02d}-Z{z_level:02d}-v0"
            )
        result.add(
            f"OmniPiano-ClairDeLune-GCFHP-{distribution}-"
            "G150-CF15-Y50-Z20-v0"
        )
        if distribution != "Shift":
            for gravity, friction in (
                ("GStep", "CFEpisode"),
                ("GEpisode", "CFStep"),
                ("GStep", "CFStep"),
            ):
                result.add(
                    f"OmniPiano-ClairDeLune-GCFHP-{gravity}-{friction}-"
                    f"{distribution}-G150-CF15-Y50-Z20-v0"
                )
    result.update(
        {
            "OmniPiano-ClairDeLune-G-Shift-N150-v0",
            "OmniPiano-ClairDeLune-CF-Shift-N15-v0",
            "OmniPiano-ClairDeLune-HP-Shift-YN50-ZN20-v0",
            "OmniPiano-ForElise-ThreeHand-Clean-v0",
        }
    )
    result.update(
        f"OmniPiano-ForElise-ThreeHand-A-Uniform-P{level:02d}-v0"
        for level in (5, 10, 15)
    )
    result.update(
        f"OmniPiano-ForElise-ThreeHand-{label}-v0"
        for label in (
            "O-Gauss-P20",
            "OR-Gauss-O20-R30",
            "ORA-Gauss-O20-R30-A10",
            "ORAG-Gauss-O20-R30-A10-G300",
            "ORAGHP-Gauss-O20-R30-A10-G300-Y25-Z10",
        )
    )
    return result


def _marl_ids() -> set[str]:
    result = {
        f"OmniPiano-{piece}-FourHand-MA-Duet-Territorial-v0"
        for piece in ("WinterWind", "PianoSonataNo301StMov", "PicturesGreatKiev")
    }
    result.update(
        f"OmniPiano-{piece}-ThreeHand-MA-MainSolo-Territorial-v0"
        for piece in (
            "WinterWind",
            "PicturesGreatKiev",
            "PolonaiseOp40No1",
            "PianoSonataNo281StMov",
        )
    )
    result.add("OmniPiano-WinterWind-FiveHand-MA-Trio-Territorial-v0")
    return result


@lru_cache(maxsize=1)
def known_single_agent_environment_ids() -> frozenset[str]:
    """Return literal and generated single-agent registration IDs without imports."""
    source = Path(__file__).resolve().parents[1] / "envs" / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    literal = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "id"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
        and keyword.value.value.startswith("OmniPiano-")
    }
    return frozenset(literal | _general_ids() | _robust_ids())


def known_marl_environment_ids() -> frozenset[str]:
    return frozenset(_marl_ids())


def _safety_choices() -> list[dict[str, Any]]:
    from omnipiano.safety.suite import EXTENSIONS, HANDS, MAIN, THRESHOLDS

    groups = {"main": MAIN, "hands": HANDS, "budget": THRESHOLDS, "extensions": EXTENSIONS}
    choices = [
        {
            "id": task.env_id,
            "label": f"{task.song} · {task.hands}H · {task.cost.semantic} / {task.cost.setting}",
            "group": group,
            "task_index": index,
            "song": task.song,
            "hands": task.hands,
            "cost": task.cost.semantic,
            "setting": task.cost.setting,
        }
        for group, tasks in groups.items()
        for index, task in enumerate(tasks)
    ]
    # Suites may intentionally expose one environment in several research
    # groups. The composer presents one canonical, unambiguous choice per ID.
    unique: dict[str, dict[str, Any]] = {}
    for choice in choices:
        unique.setdefault(choice["id"], choice)
    return list(unique.values())


def environment_catalog(track: str) -> dict[str, Any]:
    if track not in TRACKS:
        raise EnvironmentCatalogError(
            f"unknown track {track!r}; available: {', '.join(TRACKS)}"
        )
    if track == "standard":
        ids = sorted(_general_ids())
        choices = [{"id": env_id, "label": env_id} for env_id in ids]
    elif track == "robust":
        choices = [{"id": env_id, "label": env_id} for env_id in sorted(_robust_ids())]
    elif track == "safety":
        choices = _safety_choices()
    else:
        choices = [{"id": env_id, "label": env_id} for env_id in sorted(_marl_ids())]
    return {"track": track, "count": len(choices), "environments": choices}


__all__ = [
    "EnvironmentCatalogError",
    "environment_catalog",
    "known_marl_environment_ids",
    "known_single_agent_environment_ids",
]
