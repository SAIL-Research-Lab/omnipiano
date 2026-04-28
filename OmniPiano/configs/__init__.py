"""Configuration dataclasses for OmniPiano.

Centralize all user-facing configuration schemas in one place.
Keep environment construction code (make, wrappers, registry) clean by passing
typed config objects instead of many scattered keyword arguments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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
    """OmniPiano-only task variants — MJCF/XML-level modifications applied
    inside ``OmniPianoTask._apply_task_variants`` after the parent task
    builds its default model tree. These have no equivalent in the paper
    or upstream robopianist; they are pure OmniPiano extensions for
    injury / single-hand-only / morphology ablations.

    Note: paper-comparable env parameters (``disable_fingering_reward``,
    ``disable_forearm_reward``, ``disable_colorization``, etc.) live in
    ``BenchmarkEnvConfig`` — this class is intentionally restricted to
    OmniPiano-only knobs to keep the paper-vs-OmniPiano audit clean.
    """
    left_hand_immobile: bool = False
    right_hand_immobile: bool = False


@dataclass
class BenchmarkEnvConfig:
    """Shared environment defaults for all OmniPiano tasks.

    Listed in the **same order** as ``robopianist-rl/train.py:Args`` so
    each field can be diffed line-by-line against the paper recipe. All
    train.py env-related fields are mirrored here as explicit defaults
    (rather than the previous "override-only" philosophy) so that an
    experiment's env config is unambiguously visible without needing
    to walk back to upstream robopianist defaults.

    Inline-comment legend:
      * ``=``                  matches paper (run.sh-aligned where run.sh
                               overrides the train.py default; otherwise
                               train.py default = upstream default).
      * ``≠ run.sh adopts``    run.sh overrides train.py default; OmniPiano
                               adopts the run.sh value.
      * ``≠ OmniPiano-fixed``  OmniPiano permanently differs from paper
                               (with reason).

    Field consumption split (handled by ``OmniPiano.envs.registration.make``):
      * **Task kwargs** (forwarded to ``OmniPianoTask`` → ``PianoTask``):
        ``n_steps_lookahead``, ``trim_silence``, ``gravity_compensation``,
        ``reduced_action_space``, ``control_timestep``,
        ``wrong_press_termination``, ``disable_fingering_reward``,
        ``disable_forearm_reward``, ``disable_colorization``,
        ``disable_hand_collisions``, ``primitive_fingertip_collisions``,
        ``change_color_on_activation``.
      * **suite.load_with_task kwargs**: ``stretch_factor`` →
        ``stretch=``, ``shift_factor`` → ``shift=``.
      * **dm_env wrapper kwargs**: ``frame_stack`` → conditional
        ``FrameStackingWrapper``, ``clip`` → ``CanonicalSpec(clip=...)``,
        ``record_dir`` / ``record_every`` / ``record_resolution`` /
        ``camera_id`` → ``PianoSoundVideoWrapper`` (when ``record_dir``
        is set), ``action_reward_observation`` → conditional
        ``ObservationActionRewardWrapper``.

    Override resolution at make() time (later wins):
      ``BenchmarkEnvConfig`` defaults
        → ``TaskSpec.env_config`` (registry-level — single source of
          truth for experiment-defining params)
        → ``make(field=...)`` kwarg (per-field, runtime-bypass only —
          legitimate for fields that don't change experiment identity
          such as ``record_dir`` / ``camera_id`` / ``record_resolution``).

    Adding a new field — maintenance checklist
    ------------------------------------------
    The ``make()`` flow puts every field of this dataclass into ``kwargs``
    via ``setdefault`` and then routes it by either popping (wrapper /
    suite-level) or letting it fall through (task-level). When you add a
    new field here, classify it FIRST and update ``registration.py``
    accordingly:

      * **Task-level** (consumed by ``PianoWithShadowHands`` /
        ``PianoTask`` constructor): no code change needed in
        ``registration.py``. The field auto-flows into ``task_kwargs``
        and reaches the task as a constructor kwarg. Add the field
        here, list it under "Task kwargs" above, done.

      * **suite-level** (a kwarg of ``suite.load_with_task`` itself,
        like ``stretch`` / ``shift``): add a ``kwargs.pop("<field>")``
        in ``registration.py`` and pass it to ``suite.load_with_task``.
        Otherwise it falls through to task_kwargs and
        ``PianoWithShadowHands.__init__`` rejects the unknown kwarg.

      * **wrapper-level** (drives a dm_env wrapper insertion / config,
        like ``frame_stack`` / ``clip`` / ``action_reward_observation``
        / ``record_*``): add a ``kwargs.pop("<field>")`` in
        ``registration.py`` and wire it into ``_build_dm_env_chain`` at
        the correct chain position. Forgetting to pop will silently
        pass it as a task kwarg and cause a
        ``TypeError: __init__() got an unexpected keyword argument``.

    Inverse risk: if you remove a field here, also remove its
    ``kwargs.pop(...)`` in ``registration.py`` (else KeyError at make()).
    """
    # 19. n_steps_lookahead
    n_steps_lookahead: int = 10                           # ≠ run.sh adopts (train.py default 1; lets agent plan upcoming notes)
    # 20. trim_silence
    trim_silence: bool = True                             # ≠ run.sh adopts (train.py default False; episodes start with immediate activity)
    # 21. gravity_compensation
    gravity_compensation: bool = True                     # ≠ run.sh adopts (train.py default False)
    # 22. reduced_action_space
    reduced_action_space: bool = False                    # ≠ OmniPiano-fixed (run.sh: True; OmniPiano permanently False — safety tasks depend on THJ1/THJ5)
    # 23. control_timestep
    control_timestep: float = 0.05                        # = (paper-same)
    # 24. stretch_factor — passed to suite.load_with_task(stretch=...)
    stretch_factor: float = 1.0                           # = (paper-same)
    # 25. shift_factor — passed to suite.load_with_task(shift=...)
    shift_factor: int = 0                                 # = (paper-same)
    # 26. wrong_press_termination
    wrong_press_termination: bool = False                 # = (paper-same)
    # 27. disable_fingering_reward
    disable_fingering_reward: bool = False                # = (paper-same); set to True for OT-fingering tasks via env_config=BenchmarkEnvConfig(disable_fingering_reward=True) at register() time
    # 28. disable_forearm_reward
    disable_forearm_reward: bool = False                  # = (paper-same)
    # 29. disable_colorization
    disable_colorization: bool = False                    # = (paper-same)
    # 30. disable_hand_collisions
    disable_hand_collisions: bool = False                 # = (paper-same)
    # 31. primitive_fingertip_collisions
    primitive_fingertip_collisions: bool = False          # ≠ OmniPiano-fixed (run.sh: True; OmniPiano permanently False — Tasks 3 (HandCollisionConstraint) and 6 (HandCollisionForceConstraint) read contact geometry/forces directly, capsule primitives would coarsen fingertip contact boundaries and re-scale their penalties)
    # 32. frame_stack — >1 inserts FrameStackingWrapper(num_frames=frame_stack, flatten=True) into dm_env chain
    frame_stack: int = 1                                  # = (paper-same)
    # 33. clip — passed to CanonicalSpecWrapper(clip=...)
    clip: bool = True                                     # = (paper-same; clipping action to canonical [-1, 1])
    # 34. record_dir — if not None, attaches PianoSoundVideoWrapper at the dm_env layer
    record_dir: Optional[str] = None                      # = (paper-same; off by default, on for eval/render)
    # 35. record_every — frequency for PianoSoundVideoWrapper to write a clip
    record_every: int = 1                                 # = (paper-same; 1 = every episode when recording is on)
    # 36. record_resolution — (height, width); passed to PianoSoundVideoWrapper as height=, width=
    record_resolution: Tuple[int, int] = (480, 640)       # = (paper-same)
    # 37. camera_id — DmControlVideoWrapper(camera_id=...); render_checkpoint defaults to "piano/topdown" for full keyboard view
    camera_id: str = "piano/back"                         # = (paper-same)
    # 38. action_reward_observation — if True, inserts ObservationActionRewardWrapper before ConcatObs
    action_reward_observation: bool = True                # ≠ run.sh adopts (train.py default False; paper credit-assignment setup)

    # OmniPiano-only — train.py hardcodes True in get_env(), not surfaced
    # as an Args field; OmniPiano elevates to config for symmetry.
    change_color_on_activation: bool = True               # OmniPiano-only (train.py hardcoded True)

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: Optional[str] = None
    log_split: str = "train"

@dataclass
class BenchmarkProtocolConfig:
    """Shared protocol defaults for OmniPiano training + evaluation.

    Centralizes the numbers that should stay consistent across
    algorithm comparisons so learning curves and final scores are
    comparable:

      - ``total_env_steps``: total env-step interaction budget (training).
        Aligned with the RoboPianist paper's 5M-sample SAC/DroQ setup.
      - ``seed``:            scalar seed passed to the training framework.
        Both SB3 (``PPO(seed=...)`` + ``make_vec_env(seed=...)``) and
        OmniSafe (``custom_cfgs={'seed': ...}``) accept exactly one scalar
        and propagate it internally to policy init, rollout sampling, and
        ``env.reset``. Neither framework has a native concept of a list of
        seeds — paper-style replication (N independent trainings for
        mean ± std) is orchestrated by running the script N times with
        distinct seed values and aggregating the per-run eval summaries
        offline. This is the pattern used by the RoboPianist paper, SB3
        docs, and OmniSafe's ``examples/benchmarks/run_experiment_grid.py``.
        Default ``42`` follows the CleanRL convention.

        Framework default behavior without an explicit seed:
          * SB3:      ``PPO(seed=None)`` — OS entropy, NOT reproducible.
          * OmniSafe: ``PPOLag.yaml`` ships ``seed: 0`` — reproducible at
                      seed 0 unless overridden.
        We set seed explicitly so reproducibility is identical across
        frameworks rather than relying on framework-specific defaults.
      - ``num_eval_eps``:    episodes for the final benchmark eval. Passed
        to SB3 via the template's inline rollout loop and to OmniSafe via
        ``Evaluator.evaluate(num_episodes=...)``. 10 matches OmniSafe /
        D4RL eval defaults (the paper's own 1 episode is too noisy for
        mean ± std reporting).

    A single training budget is used across *all* algorithm families
    (off-policy and on-policy alike) — on-policy methods like PPO/TRPO
    will tend to underperform off-policy methods like SAC/DroQ at the
    same budget; this is intentional and rewards sample efficiency,
    matching the paper's framing.

    Not enforced in code: training scripts still accept arbitrary values.
    Treat these as the recommended defaults for producing comparable
    numbers.

    ``protocol_version`` tags reports so that if the protocol ever
    changes (e.g., bumped budget), older reports stay interpretable.

    Scope
    -----
    This class is **algorithm-agnostic by design** — every field here
    must be meaningful across SAC / PPO / DroQ / TRPO / OmniSafe-PPOLag
    alike. Algorithm-specific hparams (``batch_size``, ``discount``,
    ``replay_capacity``, ``warmstart_steps``, etc.) live in each
    trainer's argparse defaults (``examples/run_sb3_*_template.py``)
    and are dumped to ``eval_summary.json["hparams"]`` for audit. They
    do not belong here because, e.g., ``replay_capacity`` /
    ``warmstart_steps`` are off-policy-only and ``discount=0.8`` (SAC)
    vs ``0.9`` (PPO) makes a single-value default meaningless.

    Env-construction parameters (``n_steps_lookahead``, ``trim_silence``,
    ``gravity_compensation``, ``control_timestep``, ...) are mirrored
    in ``BenchmarkEnvConfig`` in train.py field order — that is the
    audit point for paper-vs-OmniPiano env diff.
    """
    total_env_steps: int = 5_000_000
    seed: int = 42
    num_eval_eps: int = 10
    protocol_version: str = "1.0"
    # TODO: Add perturbation_levels for robustness evaluation