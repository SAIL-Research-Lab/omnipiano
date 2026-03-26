import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict


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
        """Traverse wrappers to reach the underlying composer environment.

        Unwrap order:
        1) Gymnasium wrappers via `.env`
        2) Shimmy bridge via `._env`
        3) dm_env wrappers via public `.environment`
        """
        # `env` arrives as a Gymnasium wrapper layer (e.g., MetricsWrapper).
        env_ptr = env
        # Unwrap Gym wrappers first (`.env` chain).
        while hasattr(env_ptr, "env"):
            env_ptr = env_ptr.env

        # At this point we expect shimmy's DmControlCompatibilityV0 wrapper.
        if not hasattr(env_ptr, "_env"):
            raise AttributeError("Expected shimmy wrapper with a '_env' attribute.")
        dm_env = env_ptr._env

        # Prefer the public `environment` property exposed by dm_env wrappers.
        while hasattr(dm_env, "environment"):
            next_env = dm_env.environment
            if next_env is dm_env:
                break
            dm_env = next_env

        if not hasattr(dm_env, "physics") or not hasattr(dm_env, "task"):
            raise AttributeError(
                "Failed to unwrap to a composer environment exposing physics/task."
            )
        return dm_env.physics, dm_env.task


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


class HandCollisionConstraint(BaseConstraint):
    """Penalizes any collision between the two hands.

    Checks all geoms on both hands (fingers, palm, forearm, etc.).
    Cost per step = penalty_coef if any contact exists, 0 otherwise.
    """

    def __init__(self, penalty_coef: float = 1.0):
        super().__init__(penalty_coef)

    def compute_cost(self, env, action: np.ndarray, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        from mujoco_utils import collision_utils

        physics, task = self._get_dm_internals(env)
        # A `geom` is MuJoCo's collision/render geometry primitive.
        # Collect ALL geoms from each hand model (fingers, palm, forearm, etc.).
        # Use `full_identifier` (fully-qualified MJCF name) because collision_utils
        # matches contact geom names by string prefix.
        rh_geoms = [g.full_identifier for g in task.right_hand.mjcf_model.find_all("geom")]
        lh_geoms = [g.full_identifier for g in task.left_hand.mjcf_model.find_all("geom")]

        if collision_utils.has_collision(physics, rh_geoms, lh_geoms):
            return self.penalty_coef
        return 0.0

    def get_info_key(self) -> str:
        return "step_safety/cost_hand_collision"