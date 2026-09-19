"""Versioned task registration and independent main/ablation manifests."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from omnipiano.safety.semantics import CostSpec, SETTINGS, SemanticConstraint
from omnipiano.configs import BenchmarkEnvConfig


@dataclass
class SafetyEnvConfig(BenchmarkEnvConfig):
    """Expose an upstream task parameter without editing main's config class."""
    energy_penalty_coef: float = 0.0

VERSION = "safety-semantic-20260919-v1"
ALGORITHMS = ("PPO", "PPOLag", "OnCRPO", "CPO", "CUP")
SEEDS = (1, 2, 3)
BUDGETS = (1.44, 4.32, 14.4, 43.2, 144.0)
SONGS = {
    "ForElise": ("RoboPianist-repertoire-150-ForElise-v0", 399),
    "ClairDeLune": ("RoboPianist-repertoire-150-ClairDeLune-v0", 588),
    "PicturesGreatKiev": ("RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0", 720),
    "PolonaiseOp40No1": ("RoboPianist-repertoire-150-PolonaiseOp40No1-v0", 563),
}


@dataclass(frozen=True)
class Task:
    hands: int
    song: str
    cost: CostSpec
    budget: float
    layout: str = "default"

    def __post_init__(self):
        if self.hands not in (2, 3, 4, 5) or self.song not in SONGS or self.layout not in ("default", "nested"):
            raise ValueError("Invalid hands/song/layout")
        if not math.isfinite(self.budget) or self.budget < 0:
            raise ValueError("Invalid episode budget")

    @property
    def env_id(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"OmniPiano-Safety-{self.song}-{self.hands}H-{self.cost.semantic}-{self.cost.setting}-{digest}-v1"


def task(hands, song, semantic, setting, *, budget=None, layout="default", reference=None):
    protected = {2: "rh", 3: "rh", 4: "rh_t", 5: "rh_t"}[hands]
    ref = reference if reference is not None else {"actuator_power": 4 * hands,
                                                  "injured_finger": 1.0, "hand_collision": 10.0}.get(semantic, 1.0)
    cost = CostSpec(semantic, setting, reference=ref, protected_hand=protected)
    if budget is None:
        budget = round(SONGS[song][1] * (0.05 if setting in ("event", "fraction") else 0.02), 6)
    return Task(hands, song, cost, float(budget), layout)


# Retain the eight previously selected song/cost pairs; use new versioned IDs.
MAIN = (
    task(2, "ForElise", "joint_range", "fraction"),
    task(2, "ClairDeLune", "joint_range", "excess"),
    task(3, "PicturesGreatKiev", "actuator_power", "excess"),
    task(3, "PolonaiseOp40No1", "actuator_power", "excess"),
    task(4, "PicturesGreatKiev", "joint_range", "excess"),
    task(4, "PicturesGreatKiev", "actuator_power", "excess"),
    task(5, "PicturesGreatKiev", "joint_range", "event"),
    task(5, "PicturesGreatKiev", "joint_range", "excess"),
)
HANDS = tuple(task(k, "PicturesGreatKiev", "actuator_power", "excess", layout="nested",
                   reference=16, budget=14.4) for k in (2, 3, 4, 5))
BUDGET_TASK = MAIN[5]
THRESHOLDS = tuple(replace(BUDGET_TASK, budget=d) for d in BUDGETS)
# Separate candidate catalogue: 4 hand/song anchors x 11 semantic/setting cells.
EXTENSIONS = tuple(task(k, song, semantic, setting)
                   for k, song in ((2, "ForElise"), (3, "PolonaiseOp40No1"),
                                   (4, "PicturesGreatKiev"), (5, "PicturesGreatKiev"))
                   for semantic, settings in SETTINGS.items() for setting in settings)


def hand_specs(item):
    from omnipiano.tasks.hand_spec import (default_two_hand_specs, default_three_hand_specs,
                                          default_four_hand_specs, default_five_hand_specs)
    factories = {2: default_two_hand_specs, 3: default_three_hand_specs,
                 4: default_four_hand_specs, 5: default_five_hand_specs}
    if item.layout == "default":
        return factories[item.hands]()
    four = default_four_hand_specs()
    center = next(h for h in default_five_hand_specs() if h.name == "rh_c")
    return tuple([four[1], four[2], four[0], four[3], center][:item.hands])


def register_task(item):
    from omnipiano.configs import SafetyConfig
    from omnipiano.envs.registration import _registry, register
    if item.env_id in _registry:
        existing = _registry[item.env_id]
        constraints = existing.safety_config.constraints
        if len(constraints) != 1 or not isinstance(constraints[0], SemanticConstraint) or constraints[0].spec != item.cost:
            raise ValueError(f"Conflicting registration: {item.env_id}")
        return item.env_id
    register(id=item.env_id, base_env_name=SONGS[item.song][0], hand_specs=hand_specs(item),
             env_config=SafetyEnvConfig(disable_fingering_reward=True),
             safety_config=SafetyConfig(constraints=[SemanticConstraint(item.cost)]))
    return item.env_id


def register_all():
    items = {t.env_id: t for t in MAIN + HANDS + THRESHOLDS + EXTENSIONS}
    for item in items.values():
        register_task(item)
    return items


def cells(group):
    if group == "main":
        return [(t, a, s, 5_000_000) for t in MAIN for a in ALGORITHMS for s in SEEDS]
    if group == "hands":
        return [(t, a, s, 5_000_000) for t in HANDS for a in ALGORITHMS for s in SEEDS]
    if group == "budget":
        return ([(t, a, s, 5_000_000) for t in THRESHOLDS for a in ALGORITHMS[1:] for s in SEEDS]
                + [(BUDGET_TASK, "PPO", s, 5_000_000) for s in SEEDS])
    if group == "extensions":
        return [(t, a, 0, 2_000_000) for t in EXTENSIONS for a in ("PPO", "PPOLag")]
    raise ValueError(group)


def task_from_dict(value):
    value = dict(value)
    cost = dict(value.pop("cost"))
    for name in ("hand_names", "joint_names", "protected_actuators"):
        if name in cost:
            cost[name] = tuple(cost[name])
    return Task(cost=CostSpec(**cost), **value)
