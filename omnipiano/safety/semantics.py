"""Four physical safety semantics, with event/excess/fraction cost settings."""
from dataclasses import dataclass
from itertools import combinations
import math

import mujoco
import numpy as np

from omnipiano.safety.constraints import BaseConstraint

SETTINGS = {
    "joint_range": ("event", "excess", "fraction"),
    "actuator_power": ("event", "excess"),
    "injured_finger": ("event", "excess", "fraction"),
    "hand_collision": ("event", "excess", "fraction"),
}
CENTRAL_JOINTS = ("THJ2", "FFJ2", "MFJ2", "RFJ2", "LFJ2")


@dataclass(frozen=True)
class CostSpec:
    semantic: str
    setting: str
    reference: float = 1.0
    central_fraction: float = 0.5
    hand_names: tuple = ()
    joint_names: tuple = CENTRAL_JOINTS
    protected_hand: str = "rh"
    protected_actuators: tuple = ("THJ1", "THJ2", "THJ3", "THJ4", "THJ5")
    weight: float = 1.0

    def __post_init__(self):
        if self.semantic not in SETTINGS or self.setting not in SETTINGS[self.semantic]:
            raise ValueError("Unsupported semantic/setting combination")
        for name in ("reference", "central_fraction", "weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.reference <= 0 or not 0 < self.central_fraction <= 1:
            raise ValueError("reference > 0 and 0 < central_fraction <= 1 required")
        for name in ("hand_names", "joint_names", "protected_actuators"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be a tuple of unique names")
            if not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"Invalid {name}")
        if not self.joint_names or not self.protected_actuators or not self.protected_hand:
            raise ValueError("Physical targets cannot be empty")


def aggregate(excess, setting, weight=1.0):
    values = np.asarray(excess, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative per-unit excess")
    if setting == "event":
        value = float(np.any(values > 0))
    elif setting == "fraction":
        value = float(np.mean(values > 0))
    elif setting == "excess":
        value = float(np.mean(values))
    else:
        raise ValueError(setting)
    result = weight * value
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid weighted cost")
    return result


def hands_by_name(task):
    if hasattr(task, "hands_by_name"):
        return dict(task.hands_by_name)
    return {"rh": task.right_hand, "lh": task.left_hand}


def actuator_power(physics, hand):
    force = np.asarray(physics.bind(hand.actuator_force_sensors).sensordata).reshape(-1)
    speed = np.asarray(physics.bind(hand.actuator_velocity_sensors).sensordata).reshape(-1)
    if force.shape != speed.shape or force.size != len(hand.actuators):
        raise ValueError("Actuator sensor shape mismatch")
    power = np.abs(force * speed)
    if not np.isfinite(power).all():
        raise ValueError("Nonfinite actuator power")
    return power


def pair_forces(physics, hands):
    """One unit per unordered hand pair, including pairs with zero contact."""
    pairs = {pair: 0.0 for pair in combinations(range(len(hands)), 2)}
    if not pairs:
        raise ValueError("Hand collision requires at least two selected hands")
    owners = {}
    for index, hand in enumerate(hands):
        for geom in hand.mjcf_model.find_all("geom"):
            owners[int(physics.model.name2id(geom.full_identifier, "geom"))] = index
    force = np.zeros(6)
    for i, contact in enumerate(physics.data.contact):
        a, b = (owners.get(int(g)) for g in contact.geom)
        if a is None or b is None or a == b or contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(physics.model.ptr, physics.data.ptr, i, force)
        pairs[tuple(sorted((a, b)))] += abs(float(force[0]))
    values = np.asarray(list(pairs.values()))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite contact force")
    return values


class SemanticConstraint(BaseConstraint):
    """Stateless after-step measurement; compatible with main's SafetyWrapper."""

    def __init__(self, spec: CostSpec):
        super().__init__(spec.weight)
        self.spec = spec

    def compute_cost(self, env, action, obs, info):
        physics, task = self._get_dm_internals(env)
        spec = self.spec
        mapping = hands_by_name(task)
        names = spec.hand_names or tuple(mapping)
        missing = set(names) - set(mapping)
        if missing:
            raise ValueError(f"Unknown hands: {sorted(missing)}")
        hands = [mapping[name] for name in names]
        if not hands:
            raise ValueError("No hands selected")
        if spec.semantic == "joint_range":
            joints = []
            for hand in hands:
                for suffix in spec.joint_names:
                    matches = [j for j in hand.joints if j.name == suffix or j.name.endswith("_" + suffix)]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one physical joint {suffix}")
                    joints.extend(matches)
            bound = physics.bind(joints)
            ranges = np.asarray(bound.range).reshape(-1, 2)
            qpos = np.asarray(bound.qpos).reshape(-1)
            spans = ranges[:, 1] - ranges[:, 0]
            if not np.isfinite(ranges).all() or not np.isfinite(qpos).all() or np.any(spans <= 0):
                raise ValueError("Invalid physical joint positions/ranges")
            margin = (1 - spec.central_fraction) * spans / 2
            excess = np.maximum.reduce([ranges[:, 0] + margin - qpos,
                                        qpos - ranges[:, 1] + margin, np.zeros_like(qpos)]) / spans
            excess = np.where(excess > 1e-8, excess, 0.0)
            raw = excess
        elif spec.semantic == "actuator_power":
            raw = np.asarray([sum(float(actuator_power(physics, h).sum()) for h in hands)])
            excess = np.maximum(raw / spec.reference - 1, 0)
        elif spec.semantic == "injured_finger":
            if spec.protected_hand not in mapping:
                raise ValueError(f"Unknown protected hand: {spec.protected_hand}")
            hand = mapping[spec.protected_hand]
            indices = []
            for suffix in spec.protected_actuators:
                matches = [i for i, a in enumerate(hand.actuators)
                           if a.name == suffix or a.name.endswith("A_" + suffix)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one protected actuator {suffix}")
                indices.extend(matches)
            raw = actuator_power(physics, hand)[indices]
            excess = np.maximum(raw / spec.reference - 1, 0)
        else:
            raw = pair_forces(physics, hands)
            excess = np.maximum(raw / spec.reference - 1, 0)
        cost = aggregate(excess, spec.setting, spec.weight)
        prefix = f"step_safety/{spec.semantic}"
        info[prefix + "/unit_count"] = len(excess)
        info[prefix + "/raw_max"] = float(np.max(raw))
        info[prefix + "/violating_fraction"] = float(np.mean(excess > 0))
        return cost

    def get_info_key(self):
        return f"step_safety/cost_{self.spec.semantic}_{self.spec.setting}"
