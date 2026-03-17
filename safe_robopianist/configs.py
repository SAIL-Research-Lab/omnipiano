"""Configuration dataclasses for SafeRoboPianist."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from safe_robopianist.safety.constraints import BaseConstraint

@dataclass
class SafetyConfig:
    """Configuration for safety constraints."""
    constraints: List[BaseConstraint] = field(default_factory=list)
    # TODO: Add power_constraints, collision_constraints, etc.

@dataclass
class RobustConfig:
    """Configuration for robustness perturbations."""
    action_noise_std: float = 0.0
    obs_noise_std: float = 0.0
    # TODO: Add dynamics randomization configs

@dataclass
class TaskVariantConfig:
    """Configuration for task variants (e.g., single hand)."""
    left_hand_immobile: bool = False
    right_hand_immobile: bool = False
    # TODO: Add more XML/physics modifications here

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: Optional[str] = None
    log_split: str = "train"  # "train" or "eval"

@dataclass
class EvalProtocolConfig:
    """Configuration for benchmark evaluation protocol."""
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    num_eval_eps: int = 10
    # TODO: Add perturbation_levels for robustness evaluation
