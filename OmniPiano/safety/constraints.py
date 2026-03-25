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

class JointMagnitudeConstraint(BaseConstraint):
    """
    Penalizes joint actions that exceed a maximum magnitude.
    
    Note: In SafeRoboPianist, actions are typically normalized to the range [-1.0, 1.0] 
    by the `RescaleAction` wrapper before reaching this constraint. 
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