"""Configuration dataclasses for OmniPiano.

Centralize all user-facing configuration schemas in one place.
Keep environment construction code (make, wrappers, registry) clean by passing
typed config objects instead of many scattered keyword arguments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from OmniPiano.safety.constraints import BaseConstraint


@dataclass
class SafetyConfig:
    """Configuration for safety constraints."""
    # A list of instantiated safety rule objects (strategies),
    # e.g., JointMagnitudeConstraint(...). The SafetyWrapper will iterate this
    # list every step and aggregate the returned costs.
    #
    # `field(default_factory=list)` creates a NEW empty list for each SafetyConfig
    # instance. This avoids the mutable-default pitfall where multiple instances
    # would accidentally share the same list.
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
    log_split: str = "train"

@dataclass
class EvalProtocolConfig:
    """Configuration for benchmark evaluation protocol."""
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    num_eval_eps: int = 10
    # TODO: Add perturbation_levels for robustness evaluation
