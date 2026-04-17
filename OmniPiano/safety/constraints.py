import mujoco
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from OmniPiano.utils.env_unwrap import get_composer_env_from_gym

_COLLISION_MARGIN: float = 1e-8


class BaseConstraint(ABC):
    """Base class for all safety constraints."""
    
    def __init__(self, penalty_coef: float):
        self.penalty_coef = penalty_coef
        
    @abstractmethod
    def compute_cost(self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Compute the safety cost for the current step.
        Returns 0.0 if there is no violation.
        """
        pass
        
    @abstractmethod
    def get_info_key(self) -> str:
        """Return the key used to log this specific cost in the info dictionary."""
        pass

    @staticmethod
    def _get_dm_internals(env):
        """Return (physics, task) from underlying composer environment."""
        composer_env = get_composer_env_from_gym(env)
        return composer_env.physics, composer_env.task


class JointMagnitudeConstraint(BaseConstraint):
    """
    Penalizes joint actions that exceed a maximum magnitude.
    
    Note: actions are normalized to [-1.0, 1.0] by the RescaleAction wrapper.
    Therefore, `max_magnitude` should typically be a value between 0.0 and 1.0.
    """
    
    def __init__(self, index: int, max_magnitude: float, penalty_coef: float):
        super().__init__(penalty_coef)
        self.index = index
        self.max_magnitude = max_magnitude
        
    def compute_cost(self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        mag = np.abs(action[self.index])
        if mag > self.max_magnitude:
            return (mag - self.max_magnitude) * self.penalty_coef
        return 0.0
        
    def get_info_key(self) -> str:
        return f"step_safety/cost_joint_{self.index}_mag"


class MultiJointSharedMagnitudeConstraint(BaseConstraint):
    """Penalizes a group of joint actions that exceed a shared magnitude limit.

    The cost is the sum of per-joint excess magnitudes:

        sum(max(0, |action[i]| - max_magnitude) for i in indices)

    This keeps the same semantics as ``JointMagnitudeConstraint`` while making
    the restricted unit explicit at the benchmark level: a coordinated joint
    group rather than several unrelated single-joint constraints.
    """

    def __init__(
        self,
        indices: Tuple[int, ...],
        max_magnitude: float,
        penalty_coef: float,
        group_name: str,
    ):
        super().__init__(penalty_coef)
        if not indices:
            raise ValueError("indices must be non-empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"indices must be unique, got {indices}")
        if not 0.0 <= max_magnitude <= 1.0:
            raise ValueError(
                f"max_magnitude should be in [0, 1], got {max_magnitude}"
            )
        if not group_name:
            raise ValueError("group_name must be non-empty")
        self.indices = indices
        self.max_magnitude = max_magnitude
        self.group_name = group_name

    def compute_cost(
        self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> float:
        # Explicitly mark unused wrapper/context inputs
        del env, obs, info  # Unused.
        if max(self.indices) >= len(action):
            raise RuntimeError(
                f"MultiJointSharedMagnitudeConstraint indices {self.indices} exceed "
                f"action length {len(action)}"
            )
        mags = np.abs(action[list(self.indices)])
        assert len(mags) == len(self.indices), (
            f"Selected magnitude length mismatch: got {len(mags)} values for "
            f"indices {self.indices}"
        )
        excess = np.maximum(0.0, mags - self.max_magnitude)
        return self.penalty_coef * float(np.sum(excess))

    def get_info_key(self) -> str:
        return f"step_safety/cost_group_{self.group_name}_mag"


class MultiJointSummedMagnitudeConstraint(BaseConstraint):
    """Penalizes a joint group when the summed magnitudes exceed a shared budget.

    The cost is computed as:

        max(0, sum(|action[i]| for i in indices) - max_summed_magnitude)

    Unlike ``MultiJointSharedMagnitudeConstraint``, this constraint does not
    impose a per-joint ceiling. Instead, the whole motion chain shares one total
    normalized-action budget, allowing trade-offs within the group.
    """

    def __init__(
        self,
        indices: Tuple[int, ...],
        max_summed_magnitude: float,
        penalty_coef: float,
        group_name: str,
    ):
        super().__init__(penalty_coef)
        if not indices:
            raise ValueError("indices must be non-empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"indices must be unique, got {indices}")
        if not 0.0 <= max_summed_magnitude <= float(len(indices)):
            raise ValueError(
                "max_summed_magnitude should be in "
                f"[0, {len(indices)}], got {max_summed_magnitude}"
            )
        if not group_name:
            raise ValueError("group_name must be non-empty")
        self.indices = indices
        self.max_summed_magnitude = max_summed_magnitude
        self.group_name = group_name

    def compute_cost(
        self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> float:
        # This constraint depends only on the normalized action vector.
        del env, obs, info  # Unused.
        if max(self.indices) >= len(action):
            raise RuntimeError(
                f"MultiJointSummedMagnitudeConstraint indices {self.indices} "
                f"exceed action length {len(action)}"
            )
        mags = np.abs(action[list(self.indices)])
        assert len(mags) == len(self.indices), (
            f"Selected magnitude length mismatch: got {len(mags)} values for "
            f"indices {self.indices}"
        )
        summed_magnitude = float(np.sum(mags))
        excess = max(0.0, summed_magnitude - self.max_summed_magnitude)
        return self.penalty_coef * excess

    def get_info_key(self) -> str:
        return f"step_safety/cost_group_{self.group_name}_sum_mag"


class HandCollisionConstraint(BaseConstraint):
    """Penalizes any collision between the two hands.

    Checks all geoms on both hands (fingers, palm, forearm, etc.).
    Cost per step = penalty_coef if any contact exists, 0 otherwise.
    """

    def __init__(self, penalty_coef: float = 1.0):
        super().__init__(penalty_coef)
        self._rh_geoms: list = None
        self._lh_geoms: list = None

    def compute_cost(self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        from mujoco_utils import collision_utils

        physics, task = self._get_dm_internals(env)

        if self._rh_geoms is None:
            self._rh_geoms = [g.full_identifier for g in task.right_hand.mjcf_model.find_all("geom")]
            self._lh_geoms = [g.full_identifier for g in task.left_hand.mjcf_model.find_all("geom")]

        if collision_utils.has_collision(physics, self._rh_geoms, self._lh_geoms):
            return self.penalty_coef
        return 0.0

    def get_info_key(self) -> str:
        return "step_safety/cost_hand_collision"


class HandCollisionForceConstraint(BaseConstraint):
    """Uses normal contact force as continuous safety cost.

    Instead of binary collision detection (HandCollisionConstraint), this
    constraint sums the MuJoCo contact normal forces across all contact points
    between the two hands.  Light touches yield small cost; hard impacts yield
    large cost.

    Compared to binary detection, the dense force signal provides a smoother
    optimization landscape — the agent receives gradient information
    proportional to collision severity, enabling progressive learning toward
    zero-contact policies.  At the same time, this raises the difficulty for
    Lagrangian-based SafeRL methods, which must calibrate their multiplier
    against a continuously varying cost rather than a simple 0/1 boundary.

    Physics:
        The normal force at each contact point is the component perpendicular
        to the contact surface that pushes the two geoms apart.  It is the
        primary indicator of collision severity.  Extracted via
        ``mujoco.mj_contactForce()`` which returns a 6D vector
        [normal, friction1, friction2, torque1, torque2, torque3] in the
        contact frame, correctly handling both pyramidal and elliptic
        friction cone representations.
    """

    def __init__(self, penalty_coef: float = 1.0):
        super().__init__(penalty_coef)
        self._rh_geom_ids: set = None
        self._lh_geom_ids: set = None

    def _build_geom_id_cache(self, physics, task):
        """Classify every geom in the model as right-hand, left-hand, or
        neither.  Uses the same prefix-matching logic as
        ``mujoco_utils.collision_utils`` but converts results to integer ID
        sets so that per-step lookups are O(1).
        """
        rh_prefixes = [
            g.full_identifier
            for g in task.right_hand.mjcf_model.find_all("geom")
        ]
        lh_prefixes = [
            g.full_identifier
            for g in task.left_hand.mjcf_model.find_all("geom")
        ]
        self._rh_geom_ids = set()
        self._lh_geom_ids = set()
        for gid in range(physics.model.ngeom):
            name = physics.model.id2name(gid, "geom")
            if any(name.startswith(p) for p in rh_prefixes):
                self._rh_geom_ids.add(gid)
            elif any(name.startswith(p) for p in lh_prefixes):
                self._lh_geom_ids.add(gid)

        assert len(self._rh_geom_ids) == len(rh_prefixes), (
            f"RH geom count mismatch: {len(self._rh_geom_ids)} IDs found "
            f"vs {len(rh_prefixes)} MJCF prefixes"
        )
        assert len(self._lh_geom_ids) == len(lh_prefixes), (
            f"LH geom count mismatch: {len(self._lh_geom_ids)} IDs found "
            f"vs {len(lh_prefixes)} MJCF prefixes"
        )

    def compute_cost(
        self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> float:
        del action, obs  # Unused.
        physics, task = self._get_dm_internals(env)

        if self._rh_geom_ids is None:
            self._build_geom_id_cache(physics, task)

        total_normal_force = 0.0
        force_buf = np.empty(6, dtype=np.float64)
        for i, contact in enumerate(physics.data.contact):
            if contact.dist > _COLLISION_MARGIN:
                continue
            g1, g2 = contact.geom[0], contact.geom[1]
            is_hand_hand = (
                (g1 in self._rh_geom_ids and g2 in self._lh_geom_ids)
                or (g1 in self._lh_geom_ids and g2 in self._rh_geom_ids)
            )
            if is_hand_hand and contact.efc_address >= 0:
                mujoco.mj_contactForce(
                    physics.model.ptr, physics.data.ptr, i, force_buf
                )
                total_normal_force += abs(force_buf[0])

        return self.penalty_coef * total_normal_force

    def get_info_key(self) -> str:
        return "step_safety/cost_hand_collision_force"


class TotalActuatorPowerConstraint(BaseConstraint):
    """Uses raw total actuator power as dense safety cost.

    Power definition follows RoboPianist's existing energy term:
    `power = abs(force) * abs(velocity)` for each actuator, then summed over both hands.
    """

    def __init__(self, penalty_coef: float = 1.0):
        super().__init__(penalty_coef)

    def compute_cost(self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        del action, obs, info  # Unused.
        physics, task = self._get_dm_internals(env)

        right_power = task.right_hand.observables.actuators_power(physics).copy()
        left_power = task.left_hand.observables.actuators_power(physics).copy()
        total_power = float(np.sum(right_power) + np.sum(left_power))
        return self.penalty_coef * total_power

    def get_info_key(self) -> str:
        return "step_safety/cost_total_actuator_power"


class InjuredJointPowerConstraint(BaseConstraint):
    """Penalizes actuator power on specified joints to simulate injury.

    Computes ``sum(|force_i| * |velocity_i|)`` only for the injured actuators,
    reusing the same ``actuatorfrc`` / ``actuatorvel`` sensors that
    ``TotalActuatorPowerConstraint`` reads from.

    Args:
        hand: Which hand the injured joints belong to ("right" or "left").
        joint_names: Actuator suffix names (e.g. ``("WRJ1", "WRJ2")``).
            Forearm actuators use bare names (e.g. ``("forearm_tx",)``).
        penalty_coef: Multiplier applied to the summed power.
    """

    def __init__(
        self,
        hand: str,
        joint_names: Tuple[str, ...],
        penalty_coef: float = 1.0,
    ):
        super().__init__(penalty_coef)
        if hand not in ("right", "left"):
            raise ValueError(f"hand must be 'right' or 'left', got '{hand}'")
        self.hand = hand
        self.joint_names = joint_names
        self._sensor_indices: list = None

    def _build_sensor_indices(self, hand_entity):
        """Find actuator indices matching the requested joint names.

        Two naming conventions exist in the Shadow Hand model:
        - Finger/wrist actuators: ``{prefix}_A_{joint_name}`` (e.g. ``rh_A_WRJ1``)
        - Forearm actuators: bare ``{joint_name}`` (e.g. ``forearm_tx``)
        Matching uses ``endswith`` for prefixed names and exact match for bare names.
        """
        indices = []
        for i, act in enumerate(hand_entity.actuators):
            for jn in self.joint_names:
                if act.name.endswith(f"_A_{jn}") or act.name == jn:
                    indices.append(i)
                    break
        if len(indices) != len(self.joint_names):
            matched = [hand_entity.actuators[i].name for i in indices]
            raise RuntimeError(
                f"InjuredJointPowerConstraint: expected {len(self.joint_names)} "
                f"actuators for {self.joint_names}, but matched {len(indices)}: "
                f"{matched}. Available: {[a.name for a in hand_entity.actuators]}"
            )
        self._sensor_indices = indices

    def compute_cost(
        self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> float:
        del action, obs, info
        physics, task = self._get_dm_internals(env)
        hand_entity = task.right_hand if self.hand == "right" else task.left_hand

        if self._sensor_indices is None:
            self._build_sensor_indices(hand_entity)

        force = physics.bind(hand_entity.actuator_force_sensors).sensordata
        velocity = physics.bind(hand_entity.actuator_velocity_sensors).sensordata

        #check code during development
        assert len(force) == len(hand_entity.actuator_force_sensors), (
            f"Force sensor length mismatch: {len(force)} values vs "
            f"{len(hand_entity.actuator_force_sensors)} sensor elements"
        )
        assert len(velocity) == len(hand_entity.actuator_velocity_sensors), (
            f"Velocity sensor length mismatch: {len(velocity)} values vs "
            f"{len(hand_entity.actuator_velocity_sensors)} sensor elements"
        )
        assert len(force) == len(velocity) == len(hand_entity.actuators), (
            f"Actuator/sensor length mismatch: force={len(force)}, "
            f"velocity={len(velocity)}, actuators={len(hand_entity.actuators)}"
        )
        assert len(self._sensor_indices) == len(self.joint_names), (
            f"Injured actuator count mismatch: {len(self._sensor_indices)} indices "
            f"for {len(self.joint_names)} joint names {self.joint_names}"
        )
        assert all(0 <= i < len(force) for i in self._sensor_indices), (
            f"Sensor indices out of range: {self._sensor_indices} for "
            f"force/velocity length {len(force)}"
        )

        injured_power = sum(
            abs(float(force[i])) * abs(float(velocity[i]))
            for i in self._sensor_indices
        )
        return self.penalty_coef * injured_power

    def get_info_key(self) -> str:
        joints_tag = "_".join(self.joint_names).lower()
        return f"step_safety/cost_injured_{self.hand}_{joints_tag}_power"