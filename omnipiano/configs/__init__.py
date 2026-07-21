"""Configuration dataclasses for OmniPiano.

Centralize all user-facing configuration schemas in one place.
Keep environment construction code (make, wrappers, registry) clean by passing
typed config objects instead of many scattered keyword arguments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from omnipiano.safety.constraints import BaseConstraint


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
    """Configuration for robustness perturbations (v1: obs / action / reward).

    Per-distribution independent fields (Option C.3):
      - Gaussian σ:          ``{action,obs,reward}_noise_std``               (3)
      - Uniform [low, high]: ``{action,obs,reward}_noise_uniform_{low,high}``(6)
      - Shift constant:      ``{action,obs,reward}_noise_shift``             (3)
      - Distribution knob:   ``noise_dist``                                  (1)
    Total: 13 fields. Each field name maps strictly to its distribution's
    mathematical parameter, mirroring Robust-Gymnasium's ``--noise-sigma`` /
    ``--uniform-low/high`` / ``--noise-shift`` parameterization.

    This config defines the TRAINING-time noise only — it is part of the
    registered task's identity. The eval-time noise multiplier is NOT a
    field here (2026-07-21 decision): it is a measurement parameter, not
    an experiment-identity parameter, and lives as a public ``make()``
    kwarg instead — ``omnipiano.make(env_id, mode="eval",
    eval_noise_scale=...)`` (default 1.0 = matched eval; 0.0 = clean;
    >1 = stress). All three channels INCLUDING reward scale identically
    (decision 11: matched at eval, never force-zeroed).

    Uniform bounds use their NATURAL parameters (Option 4a) — **no**
    std-matching conversion:
      - Symmetric (v1 default registrations): ``low = -level, high = +level``.
      - Asymmetric (advanced users may register): e.g.
        ``low=-0.02, high=+0.10`` to model biased sensor drift.
    ``noise_dist`` is a single scalar shared across channels (one noise type
    per registered env, mirroring RG's ``--noise-type``); per-channel distinct
    distributions are not supported in v1.

    ``__post_init__`` validates uniform bounds (finite, low <= high) and
    raises if a channel sets fields inconsistent with ``noise_dist``.
    """

    # === Gaussian σ (per channel) ===
    action_noise_std: float = 0.0
    obs_noise_std: float = 0.0
    reward_noise_std: float = 0.0

    # === Uniform [low, high] (per channel, allows asymmetric) ===
    action_noise_uniform_low: float = 0.0
    action_noise_uniform_high: float = 0.0
    obs_noise_uniform_low: float = 0.0
    obs_noise_uniform_high: float = 0.0
    reward_noise_uniform_low: float = 0.0
    reward_noise_uniform_high: float = 0.0

    # === Shift constant offset (per channel) ===
    action_noise_shift: float = 0.0
    obs_noise_shift: float = 0.0
    reward_noise_shift: float = 0.0

    # === Distribution selector (shared across channels) ===
    noise_dist: Literal["gaussian", "uniform", "shift"] = "gaussian"

    def __post_init__(self):
        """Validate config integrity (raises ``ValueError`` on violation):

        0. noise_dist is a known value; stds are finite and non-negative,
           shifts are finite — fail fast rather than let a typo silently
           produce a clean env or crash mid-episode (NaN passes any
           ``< 0`` check, so finiteness must be tested explicitly).
        1. Uniform bounds are finite and ordered (``low <= high``).
        2. A channel does not set fields inconsistent with ``noise_dist``
           (e.g. ``noise_dist='gaussian'`` but a ``*_noise_shift`` is
           nonzero) — catches accidental mis-registration at construction
           time rather than letting it silently drift.
        """
        valid_dists = ("gaussian", "uniform", "shift")
        if self.noise_dist not in valid_dists:
            raise ValueError(
                f"RobustConfig: noise_dist must be one of {valid_dists}, "
                f"got {self.noise_dist!r}"
            )
        for ch in ("action", "obs", "reward"):
            std = getattr(self, f"{ch}_noise_std")
            if not np.isfinite(std):
                raise ValueError(
                    f"RobustConfig: {ch}_noise_std must be finite, got {std}"
                )
            if std < 0.0:
                raise ValueError(
                    f"RobustConfig: {ch}_noise_std must be >= 0 (standard "
                    f"deviation), got {std}"
                )
            # Shift may legitimately be negative (e.g. A-Shift-N15 = -0.15),
            # so only finiteness is enforced.
            shift = getattr(self, f"{ch}_noise_shift")
            if not np.isfinite(shift):
                raise ValueError(
                    f"RobustConfig: {ch}_noise_shift must be finite, "
                    f"got {shift}"
                )

        for ch in ("action", "obs", "reward"):
            lo = getattr(self, f"{ch}_noise_uniform_low")
            hi = getattr(self, f"{ch}_noise_uniform_high")
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(
                    f"RobustConfig: {ch}_noise_uniform bounds must be finite, "
                    f"got low={lo}, high={hi}"
                )
            if lo > hi:
                raise ValueError(
                    f"RobustConfig: {ch}_noise_uniform_low ({lo}) must be "
                    f"<= high ({hi})"
                )

        active_dist = self.noise_dist
        for ch in ("action", "obs", "reward"):
            std_set = getattr(self, f"{ch}_noise_std") != 0.0
            unif_set = (
                getattr(self, f"{ch}_noise_uniform_low") != 0.0
                or getattr(self, f"{ch}_noise_uniform_high") != 0.0
            )
            shift_set = getattr(self, f"{ch}_noise_shift") != 0.0

            if active_dist == "gaussian" and (unif_set or shift_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='gaussian' but "
                    f"{ch}_noise_uniform_* or {ch}_noise_shift is nonzero. "
                    f"Use {ch}_noise_std instead, or set noise_dist to match."
                )
            if active_dist == "uniform" and (std_set or shift_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='uniform' but "
                    f"{ch}_noise_std or {ch}_noise_shift is nonzero. "
                    f"Use {ch}_noise_uniform_low/high instead."
                )
            if active_dist == "shift" and (std_set or unif_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='shift' but "
                    f"{ch}_noise_std or {ch}_noise_uniform_* is nonzero. "
                    f"Use {ch}_noise_shift instead."
                )

    # ------------------------------------------------------------------
    # Noise semantics — single source of truth, used by BOTH the gym-layer
    # RobustWrapper (action / reward channels) and the dm_env-layer
    # DmEnvObsNoiseWrapper (obs channel), so the two layers never diverge.
    # ------------------------------------------------------------------
    def is_channel_active(self, channel: str) -> bool:
        """True if ``channel`` has nonzero noise magnitude under noise_dist
        (gaussian→std, uniform→low/high, shift→shift)."""
        if self.noise_dist == "gaussian":
            return getattr(self, f"{channel}_noise_std") != 0.0
        if self.noise_dist == "uniform":
            return (getattr(self, f"{channel}_noise_uniform_low") != 0.0
                    or getattr(self, f"{channel}_noise_uniform_high") != 0.0)
        if self.noise_dist == "shift":
            return getattr(self, f"{channel}_noise_shift") != 0.0
        raise ValueError(f"Unknown noise_dist: {self.noise_dist!r}")

    def sample_noise(self, rng, channel: str, shape):
        """Sample noise for ``channel`` from the active distribution.

        - gaussian: read ``{channel}_noise_std`` as σ → N(0, σ²) per-dim
          (step-level). Kept **bit-identical** to a direct
          ``rng.normal(0, std, size=shape)`` call (equivalence gate §0.7).
        - uniform:  read ``{channel}_noise_uniform_{low,high}`` → U[low, high]
          per-dim (step-level); bounds may be asymmetric.
        - shift:    read ``{channel}_noise_shift`` → constant offset broadcast
          to all dims; **NO rng draw** (deterministic, program-run-level).
        """
        if self.noise_dist == "gaussian":
            std = getattr(self, f"{channel}_noise_std")
            return rng.normal(0.0, std, size=shape)
        if self.noise_dist == "uniform":
            lo = getattr(self, f"{channel}_noise_uniform_low")
            hi = getattr(self, f"{channel}_noise_uniform_high")
            return rng.uniform(lo, hi, size=shape)
        if self.noise_dist == "shift":
            shift = getattr(self, f"{channel}_noise_shift")
            return np.full(shape, shift, dtype=float)
        raise ValueError(f"Unknown noise_dist: {self.noise_dist!r}")

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

    Field consumption split (handled by ``omnipiano.envs.registration.make``):
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
      - ``num_eval_eps``:    episodes per eval call. Default 1 matches the
        RoboPianist paper convention ("we evaluate the F1 every 10K
        training steps for 1 episode (no stochasticity in the
        environment)"). Verified empirically: OmniPiano leaves
        ``_randomize_hand_positions=False`` and ``RobustConfig.*_noise_std=0.0``
        by default, and mujoco physics is deterministic given fixed init
        state + actions. The ~0.05% per-episode variance we observe is a
        numerical-noise floor (mujoco SIMD + torch threading reductions
        not order-stable), much smaller than algorithm-level effects.
        Cross-RUN variance (3 training seeds) is where paper-level
        confidence intervals come from, not within-run multi-episode
        eval. SB3 path reads this via the template's argparse default;
        OmniSafe path passes it to our custom eval loop in
        ``run_omnisafe_template._final_eval`` (we bypass
        ``Evaluator.evaluate`` to add seed control + surface
        ``episode_task/f1`` from the terminal info dict).
      - ``eval_freq_env_steps``: periodic-evaluation cadence in env-steps,
        unified across all algorithm backends so learning curves overlay
        cleanly. SB3 EvalCallback consumes this directly as ``--eval-freq``.
        OmniSafe (whose ckpt frequency is in *epochs*, not env-steps)
        derives ``save_model_freq = ceil(eval_freq_env_steps /
        steps_per_epoch)``. Each framework snaps to its natural rollout/
        epoch boundary, so the realized interval is ≥ this value:
        SB3 PPO @ n_envs=16, n_steps=2048 → effective ~65,536 env-steps;
        OmniSafe PPOLag @ steps_per_epoch=20,000, save_model_freq=3
        → 60,000 env-steps. Within ±10K of each other — well inside
        plotting tolerance.
      - ``gamma``:           discount factor unified across ALL algorithms.
        Treated as a *task property*, not an algorithm hparam: piano control
        has empirically short effective horizon (~5 steps) and gamma encodes
        "how many steps to credit-assign". RoboPianist (2023) sets
        ``--discount 0.8`` in ``robopianist-rl/run.sh``; OmniPiano adopts the
        same value uniformly for SAC, PPO, TQC and all future baselines.
        Independently validated on 3-hand PicturesGreatKiev Prototype:
        SAC peak F1 0.235 → 0.422 (+18.7 pts) and PPO mean peak F1 0.195
        → 0.401 (+20.6 pts) at gamma 0.99 → 0.8, single-variable ablation.
        Off-policy and on-policy methods both benefit, supporting the
        task-property framing. Following D4RL / dm_control convention,
        task-specific hparams are unified across algorithms while algo-
        specific hparams (``batch_size``, ``learning_starts``,
        ``target_entropy``, ...) stay at each library's default for
        defensibility to reviewers.

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
    alike. Algorithm-specific hparams (``batch_size``,
    ``replay_capacity``, ``warmstart_steps``, ``target_entropy``, ...)
    live in each trainer's argparse defaults
    (``examples/run_sb3_*_template.py``) and are dumped to
    ``eval_summary.json["hparams"]`` for audit. They do not belong here
    because they are either off-policy-only (``replay_capacity``,
    ``warmstart_steps``) or SAC-family-only (``target_entropy``) with no
    cross-algorithm equivalent.

    ``gamma`` is the one historical exception: although it is a standard
    algorithm hparam, in OmniPiano it is treated as a *task* property
    (see the ``gamma`` field doc above) and therefore lives here.

    Env-construction parameters (``n_steps_lookahead``, ``trim_silence``,
    ``gravity_compensation``, ``control_timestep``, ...) are mirrored
    in ``BenchmarkEnvConfig`` in train.py field order — that is the
    audit point for paper-vs-OmniPiano env diff.
    """
    total_env_steps: int = 5_000_000
    seed: int = 42
    num_eval_eps: int = 1
    gamma: float = 0.8
    eval_freq_env_steps: int = 50_000
    protocol_version: str = "1.0"
    # TODO: Add perturbation_levels for robustness evaluation