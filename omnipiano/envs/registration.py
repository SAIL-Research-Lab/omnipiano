"""Environment registration and factory for OmniPiano (paper-chain version).

This module implements register() + make() pattern. The `make()` function
builds the **paper-chain** env layout — replicating
``robopianist-rl/train.py:get_env()`` exactly at the dm_env layer, then
attaching OmniPiano's modular gym wrappers on top. This produces flat
Box observations matching paper-trained policies (MlpPolicy), unlike the
prior shimmy/Dict-obs chain which underperformed paper benchmarks.

dm_env layer:
    suite.load_with_task(OmniPianoTask, stretch=, shift=)
    → EpisodeStatisticsWrapper
    → [PianoSoundVideoWrapper if record_dir]
    → MidiEvaluationWrapper
    → [DmEnvObsNoiseWrapper if robust_config obs channel active]
    → [ObservationActionRewardWrapper if action_reward_observation]
    → ConcatObservationWrapper           # Dict → flat ndarray
    → [FrameStackingWrapper if frame_stack > 1]
    → CanonicalSpecWrapper(clip=clip)    # action canonical [-1, 1]
    → SinglePrecisionWrapper             # obs → float32

gym boundary:
    DmEnvToGymnasium  (custom adapter — no shimmy, no monkey-patch)

gym layer (OmniPiano modular wrappers):
    → MetricsWrapper           # task reward terms + episode-end musical metrics → info
    → SafetyWrapper            # constraint costs → info
    → RobustWrapper            # action noise (gym layer) + obs noise reporting
    → [SafeRecordEpisodeStatistics if mode == "eval"]

See `examples/run_sb3_template.py` for a typical SB3 caller.
"""

# ---------------------------------------------------------------------------
# numpy 2.0 compat shim — must run BEFORE any dm_env_wrappers import.
#
# `dm_env_wrappers.SinglePrecisionWrapper` (and a few others) call
# ``np.array(value, copy=False)``. numpy 2.0 made this strict: it raises
# ValueError when a copy would otherwise be required, instead of falling
# through to copying as in numpy 1.x. We intercept ``np.array`` at this
# module's import time and route ``copy=False`` calls to ``np.asarray``,
# which preserves numpy 1.x's "copy only if necessary" semantics.
#
# This is **a permanent shim, not a stopgap**. Upstream
# ``kevinzakka/dm_env_wrappers`` has been effectively abandoned:
#   - Last PyPI release v0.0.13 was 2023-10-05; no newer release.
#   - ``single_precision.py`` last touched 2022-10-13.
#   - No open PRs, no issues mentioning numpy 2.0 (verified 2026-04-24).
#   - Active forks (garymm, fermex2003-cyber) also haven't fixed it.
# So waiting for an upstream fix is not realistic. The shim stays until
# we either (a) vendor + patch the ~5 wrappers we actually use, or
# (b) hand-write replacements (each is < 50 lines).
#
# Applied here at registration import time so all callers of
# omnipiano.make() benefit without each script re-applying the patch.
# Subprocess fork inherits the patched ``np.array`` via COW; spawn
# subprocesses re-import this module and re-apply.
# ---------------------------------------------------------------------------
import numpy as _np

_orig_np_array = _np.array


def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
        return _np.asarray(*args, **kwargs)
    return _orig_np_array(*args, **kwargs)


_np.array = _np_array_compat  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Sequence

from dm_env_wrappers import (
    CanonicalSpecWrapper,
    ConcatObservationWrapper,
    EpisodeStatisticsWrapper,
    FrameStackingWrapper,
    ObservationActionRewardWrapper,
    SinglePrecisionWrapper,
)
from robopianist import suite
from robopianist.wrappers.evaluation import MidiEvaluationWrapper

from omnipiano.configs import (
    BenchmarkEnvConfig,
    RobustConfig,
    SafetyConfig,
    TaskVariantConfig,
)
from omnipiano.envs.dm_env_adapter import DmEnvToGymnasium
from omnipiano.envs.dm_env_obs_noise import (
    DmEnvObsNoiseWrapper,
    OBS_NOISE_SEED_OFFSET,
)
from omnipiano.tasks.hand_spec import HandSpec
from omnipiano.tasks.omni_piano_task import OmniPianoTask
from omnipiano.wrappers.metrics_wrapper import MetricsWrapper
from omnipiano.wrappers.robust_wrapper import RobustWrapper
from omnipiano.wrappers.safety_wrapper import SafetyWrapper


# ---------------------------------------------------------------------------
# Caller-facing kwargs whitelist
# ---------------------------------------------------------------------------
# Only these may be passed via ``make(env_name, **kwargs)``. They are
# strictly *runtime-bypass* — none affects the environment's training
# trajectory, so legitimately varying them per call doesn't change the
# experiment's identity:
#
#   * ``seed``               — per-call replication seed (different
#                              seeds = different replications of the
#                              same registered experiment).
#   * ``record_dir`` /
#     ``record_every`` /
#     ``record_resolution`` /
#     ``camera_id``          — eval-time video recording knobs; the
#                              policy doesn't see any of these.
#
# Everything else (``n_steps_lookahead``, ``control_timestep``,
# ``disable_fingering_reward``, ``hand_specs``, ``midi_file``,
# ``frame_stack``, ``action_reward_observation``, ...) MUST come from
# the registered ``TaskSpec``. Want a different value? ``register()``
# a new task id. This makes ``env_name`` the single canonical handle
# for every experiment-defining choice — ``eval_summary.json`` only
# needs to record the env id to fully describe the run.
_RUNTIME_BYPASS_FIELDS = frozenset({
    "seed",
    "record_dir",
    "record_every",
    "record_resolution",
    "camera_id",
})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    """Specification for a registered benchmark task."""
    base_env_name: str
    safety_config: Optional[SafetyConfig] = None
    robust_config: Optional[RobustConfig] = None
    task_config: Optional[TaskVariantConfig] = None
    env_config: Optional[BenchmarkEnvConfig] = None
    # N-hand morphology. None = use PianoTask's default 2-hand (rh, lh) pair.
    hand_specs: Optional[Sequence[HandSpec]] = None


_registry: Dict[str, TaskSpec] = {}


def register(id: str, **kwargs):
    """Register a benchmark task by name.

    Args:
        id: Unique task identifier, e.g. "OmniPiano-ForElise-WristLimit-v0".
        **kwargs: Fields of TaskSpec (base_env_name, safety_config,
            robust_config, task_config, env_config, hand_specs).
    """
    _registry[id] = TaskSpec(**kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make(
    env_name: str,
    log_dir: Optional[str] = None,
    mode: str = "train",
    **kwargs,
):
    """Build a paper-chain OmniPiano gymnasium env.

    Args:
        env_name: Registered OmniPiano task id (e.g.
            ``"OmniPiano-ForElise-FingeringOT-v0"``). MUST be present in
            the registry — unknown ids raise ``ValueError`` with a list
            of valid ids. All four configs (``robust_config`` /
            ``safety_config`` / ``task_config`` / ``env_config``) are
            sourced exclusively from the registry; to vary any of
            them, register a new task id rather than overriding at
            make() time. This keeps every experiment configuration
            tied to a canonical, traceable id.
        log_dir: If set, eval mode writes per-episode CSV here.
        mode: ``"train"`` or ``"eval"`` (renamed from ``log_split``,
            decision 9 — one flag selecting the env's train/eval regime).
            At ``"eval"``: all RobustConfig magnitude fields are scaled by
            ``eval_noise_scale`` (0.0=clean default, 1.0=training-level,
            >1=stress) and reward noise is force-zeroed (reward perturbation
            is training-only, §0.6.3); additionally, if ``log_dir`` is set,
            SafeRecordEpisodeStatistics is attached. At ``"train"`` the noise
            scale is 1.0 (registered training noise).
        **kwargs: Strictly whitelisted to ``_RUNTIME_BYPASS_FIELDS``
            (``seed``, ``record_dir``, ``record_every``,
            ``record_resolution``, ``camera_id``) — none of which
            affects the policy's observation/training trajectory. Any
            other key raises ``ValueError`` directing the caller to
            register a new task id instead.

    Returns:
        gymnasium.Env. Observation space is flat ``Box(shape=(N,), dtype=float32)``;
        action space is ``Box([-1, 1]^M, dtype=float32)``. info dict at
        each step contains ``task/*``, ``step_safety/*``, ``robust/*``;
        at episode termination also ``episode_task/*`` and
        ``episode_safety/*``.
    """
    # ------------------------------------------------------------------
    # 0a. Multi-agent env_id redirect — give a clear error if the user
    #     accidentally passed a `-MA-` env id to the single-agent factory.
    #     Per ma_territorial_impl_plan.md § 1: SA and MA are two distinct
    #     entry points (gymnasium.Env vs PettingZoo ParallelEnv have
    #     incompatible step signatures); silent failure is unhelpful, so
    #     fail fast with a specific redirect to omnipiano.make_parallel().
    # ------------------------------------------------------------------
    if "-MA-" in env_name:
        raise ValueError(
            f"env_id {env_name!r} is a multi-agent env (contains '-MA-'). "
            f"omnipiano.make() returns a gymnasium.Env (single-agent); "
            f"use omnipiano.make_parallel({env_name!r}, ...) instead, "
            f"which returns a PettingZoo ParallelEnv."
        )

    # ------------------------------------------------------------------
    # 0b. Registry lookup — resolve configs from TaskSpec.
    #    All 4 configs are registry-only (single source of truth).
    #    Caller-side runtime bypass for env-construction fields like
    #    ``record_dir`` / ``camera_id`` flows through ``**kwargs`` and
    #    overrides env_config defaults via setdefault below.
    # ------------------------------------------------------------------
    if env_name not in _registry:
        # Fail-fast on unknown ids. Without this check, an unregistered
        # name would silently flow into ``suite.load_with_task`` which
        # raises a confusing error listing only RoboPianist envs (none
        # of the OmniPiano-* ids), masking typos in the OmniPiano id.
        raise ValueError(
            f"Unknown OmniPiano env id: {env_name!r}. "
            f"Register it first in omnipiano/envs/__init__.py via "
            f"register(id=..., base_env_name='RoboPianist-...'). "
            f"Currently registered ({len(_registry)}): "
            f"{sorted(_registry.keys())}"
        )

    # Enforce the runtime-bypass-only whitelist on caller kwargs.
    # Anything else (including trajectory-affecting env_config fields,
    # ``hand_specs``, ``midi_file``) must be set via the registry.
    illegal = set(kwargs) - _RUNTIME_BYPASS_FIELDS
    if illegal:
        raise ValueError(
            f"make(**kwargs) only accepts runtime-bypass fields "
            f"{sorted(_RUNTIME_BYPASS_FIELDS)}. Got illegal overrides: "
            f"{sorted(illegal)}. To vary these, register a new task id "
            f"in omnipiano/envs/__init__.py."
        )

    task_spec = _registry[env_name]
    base_env_name = task_spec.base_env_name
    robust_config = task_spec.robust_config or RobustConfig()
    safety_config = task_spec.safety_config or SafetyConfig()
    task_config = task_spec.task_config or TaskVariantConfig()
    env_config = task_spec.env_config or BenchmarkEnvConfig()
    hand_specs = task_spec.hand_specs

    # Apply env_config defaults to kwargs. Caller's runtime-bypass
    # kwargs (e.g., record_dir override) win via setdefault.
    for k, v in dataclasses.asdict(env_config).items():
        kwargs.setdefault(k, v)

    # ------------------------------------------------------------------
    # 0c. Effective robust config — mode / eval_noise_scale plumbing (§0.2).
    #     At mode=="eval", every RobustConfig magnitude field (all three
    #     channels, INCLUDING reward) is multiplied by eval_noise_scale
    #     (1.0=matched default, 0.0=clean/nominal, >1=stress); at
    #     mode=="train" the scale is 1.0 (registered training noise).
    #
    #     Reward is treated symmetrically with action/obs (decision 11): it is
    #     NOT force-zeroed at eval. Because ObservationActionRewardWrapper
    #     feeds the (noised) reward back into obs["reward"] — a policy input —
    #     suppressing reward noise at eval would push obs["reward"] out of the
    #     training distribution. Matched eval keeps it in-distribution; the
    #     true (denoised) return is recovered as a separate CSV column
    #     (ep_return_true = sum of reward-decomposition terms, §0.6), not by
    #     killing the noise. Everything downstream (RobustWrapper,
    #     DmEnvObsNoiseWrapper) consumes effective_robust_config; the raw
    #     registered robust_config is kept unmodified for reproducibility.
    # ------------------------------------------------------------------
    scale = robust_config.eval_noise_scale if mode == "eval" else 1.0
    effective_robust_config = dataclasses.replace(
        robust_config,
        action_noise_std=robust_config.action_noise_std * scale,
        obs_noise_std=robust_config.obs_noise_std * scale,
        reward_noise_std=robust_config.reward_noise_std * scale,
        action_noise_uniform_low=robust_config.action_noise_uniform_low * scale,
        action_noise_uniform_high=robust_config.action_noise_uniform_high * scale,
        obs_noise_uniform_low=robust_config.obs_noise_uniform_low * scale,
        obs_noise_uniform_high=robust_config.obs_noise_uniform_high * scale,
        reward_noise_uniform_low=robust_config.reward_noise_uniform_low * scale,
        reward_noise_uniform_high=robust_config.reward_noise_uniform_high * scale,
        action_noise_shift=robust_config.action_noise_shift * scale,
        obs_noise_shift=robust_config.obs_noise_shift * scale,
        reward_noise_shift=robust_config.reward_noise_shift * scale,
    )

    # ------------------------------------------------------------------
    # 1. Extract suite/wrapper-level fields from kwargs.
    #    The rest is task-level and becomes task_kwargs (forwarded to
    #    OmniPianoTask → PianoWithShadowHands constructor).
    # ------------------------------------------------------------------
    seed = kwargs.pop("seed", None)

    # suite.load_with_task kwargs
    stretch = kwargs.pop("stretch_factor")
    shift = kwargs.pop("shift_factor")

    # dm_env wrapper kwargs
    frame_stack = kwargs.pop("frame_stack")
    clip = kwargs.pop("clip")
    record_dir = kwargs.pop("record_dir")
    record_every = kwargs.pop("record_every")
    record_height, record_width = kwargs.pop("record_resolution")
    camera_id = kwargs.pop("camera_id")
    action_reward_observation = kwargs.pop("action_reward_observation")

    # Whatever remains in `kwargs` is task-level — forwarded to OmniPianoTask
    # below as task_kwargs. Snapshot so closure rebuilds don't re-pop.
    task_kwargs_base = dict(kwargs)

    # ------------------------------------------------------------------
    # 2. dm_env builder closure — called by the adapter at init AND on
    #    reset(seed=X) whenever the new seed differs from the current
    #    one.
    #
    #    Rebuild (not in-place reseed) is required because dm_control's
    #    ``composer_utils.Environment`` fixes ``random_state`` at
    #    construction time and exposes no setter. To honor gymnasium's
    #    ``Env.reset(seed=X)`` contract — that the env must actually
    #    reseed its RNG — we rebuild the entire dm_env chain.
    #
    #    This applies uniformly to any framework respecting gym.Env
    #    semantics: SB3 ``make_vec_env``, OmniSafe vector envs,
    #    gymnasium's ``AsyncVectorEnv``, CleanRL single-env runs, rllib
    #    RolloutWorker, etc. All set per-sub-env / per-replication seeds
    #    via ``reset(seed=...)`` and benefit equally.
    # ------------------------------------------------------------------
    def _build_dm_env_chain(_seed: Optional[int]):
        task_kwargs = {"task_config": task_config, **task_kwargs_base}
        if hand_specs is not None:
            task_kwargs["hand_specs"] = hand_specs
        env = suite.load_with_task(
            environment_name=base_env_name,
            task_cls=OmniPianoTask,
            midi_file=None,
            seed=_seed,
            stretch=stretch,
            shift=shift,
            task_kwargs=task_kwargs,
        )
        env = EpisodeStatisticsWrapper(env, deque_size=1)

        if record_dir is not None:
            from robopianist.wrappers.sound import PianoSoundVideoWrapper
            env = PianoSoundVideoWrapper(
                env,
                record_dir=record_dir,
                record_every=record_every,
                camera_id=camera_id,
                height=record_height,
                width=record_width,
            )

        # MidiEvaluationWrapper is required by MetricsWrapper's
        # episode-end branch (it calls find_dm_env_wrapper(dm_env,
        # MidiEvaluationWrapper)). Always include it.
        env = MidiEvaluationWrapper(env, deque_size=1)

        # Per-key obs noise (only when ObservationRobust task requests it).
        # Must come BEFORE ConcatObservationWrapper so it can pick keys
        # by name (skip categorical/counter keys like "goal").
        if effective_robust_config.is_channel_active("obs"):
            # Derive obs-noise RNG seed from the master seed but with an
            # offset so the noise stream is independent of the task's
            # internal random_state (used for episode init).
            obs_noise_seed = (
                (_seed + OBS_NOISE_SEED_OFFSET) if _seed is not None else None
            )
            env = DmEnvObsNoiseWrapper(
                env,
                robust_config=effective_robust_config,
                seed=obs_noise_seed,
            )

        # Paper "--action-reward-observation": prev_action + prev_reward
        # into obs dict for credit assignment. Must come BEFORE ConcatObs
        # (operates on Dict).
        if action_reward_observation:
            env = ObservationActionRewardWrapper(env)

        # Dict → flat ndarray. From here on obs has no key structure.
        env = ConcatObservationWrapper(env)

        # Frame stacking — operates on the flat obs vector. Must come
        # AFTER ConcatObservationWrapper. Skipped at frame_stack=1
        # because FrameStackingWrapper still wraps a deque internally
        # and adds zero value when num_frames=1.
        if frame_stack > 1:
            env = FrameStackingWrapper(
                env, num_frames=frame_stack, flatten=True
            )

        # Action canonical [-1, 1] with optional clipping. When clip=True
        # the agent-side [-1,1] action is clipped then rescaled to the
        # actuator's physical range before reaching physics.
        env = CanonicalSpecWrapper(env, clip=clip)

        # Cast obs to float32 (dm_env's MuJoCo observables default to
        # float64). Saves 2x IPC bandwidth across SubprocVecEnv and
        # matches paper's training dtype.
        env = SinglePrecisionWrapper(env)

        return env

    # ------------------------------------------------------------------
    # 3. dm_env → gymnasium boundary
    # ------------------------------------------------------------------
    gym_env = DmEnvToGymnasium(env_builder=_build_dm_env_chain, seed=seed)

    # ------------------------------------------------------------------
    # 4. gym layer — OmniPiano modular wrappers
    #    Order: Metrics (innermost) → Safety (middle) → Robust (outermost)
    # ------------------------------------------------------------------
    env = MetricsWrapper(gym_env)
    env = SafetyWrapper(env, config=safety_config)
    env = RobustWrapper(env, config=effective_robust_config)

    # ------------------------------------------------------------------
    # 5. Optional eval CSV logger
    # ------------------------------------------------------------------
    if log_dir is not None and mode == "eval":
        from omnipiano.utils.logger_wrapper import SafeRecordEpisodeStatistics
        env = SafeRecordEpisodeStatistics(
            env,
            log_dir=log_dir,
            split=mode,
        )

    return env
