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
    """Configuration for task variants (e.g., single hand, injury simulation).

    Fields that map directly to PianoWithShadowHands constructor kwargs
    (like disable_fingering_reward) are forwarded at task construction time.
    """
    left_hand_immobile: bool = False
    right_hand_immobile: bool = False
    disable_fingering_reward: bool = False


@dataclass
class BenchmarkEnvConfig:
    """Shared environment defaults for all OmniPiano tasks.

    Only parameters where OmniPiano *intentionally overrides* the upstream
    robopianist defaults are listed here. Parameters whose OmniPiano value
    would equal the upstream default are deliberately omitted — they
    inherit upstream via kwarg fall-through, avoiding tautological
    restatement and the drift risk of pinning "the same" value twice.

    Any task may override any field by passing a customized instance to
    TaskSpec.env_config; any caller may override any field at make() time
    via **kwargs. No field is special.
    """
    # Physics fidelity (upstream defaults both False).
    gravity_compensation: bool = True
    # primitive_fingertip_collisions: bool = True
    # Deliberately left at upstream default (False = true fingertip meshes).
    # Task 3 (HandCollisionConstraint) and Task 6 (HandCollisionForceConstraint)
    # read contact geometry/forces directly; capsule primitives would coarsen
    # fingertip contact boundaries and alter normal-force magnitudes,
    # implicitly re-scaling their penalties. Keep exact meshes so collision
    # safety signals stay faithful. If training throughput later proves to be
    # a bottleneck, re-enable here AND add an explicit per-task override
    # (env_config=BenchmarkEnvConfig(primitive_fingertip_collisions=False))
    # on Tasks 3 and 6 so collision fidelity stays pinned where it matters.

    # Observation horizon — upstream default is 1 step; 10 steps gives the
    # agent enough lookahead to plan hand motion for upcoming notes.
    n_steps_lookahead: int = 10

    # MIDI preprocessing — upstream default is False; trimming leading
    # silence makes episodes start with immediate agent activity.
    trim_silence: bool = True

    # Visualization — upstream default is False; highlighting activated
    # keys is universally enabled in existing OmniPiano usage.
    change_color_on_activation: bool = True

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
