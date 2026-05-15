"""MA env registry + factory.

Mirrors ``OmniPiano/envs/registration.py`` (SA side) but builds a PettingZoo
``ParallelEnv`` instead of a Gymnasium ``Env``. Reuses the SA dm_env chain
1:1 except for ``ConcatObservationWrapper`` (the only wrapper that's
fundamentally incompatible with Dict obs).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import dm_env
import numpy as np

from OmniPiano.configs import (
    BenchmarkEnvConfig,
    RobustConfig,
    SafetyConfig,
    TaskVariantConfig,
)
from OmniPiano.multiagent.assignment import (
    AGENT_ASSIGNMENTS,
    MorphologyAssignment,
)
from OmniPiano.multiagent.parallel_env import OmniPianoParallelEnv


# Mirror SA `make()`'s runtime-bypass whitelist (these don't affect training trajectory).
_RUNTIME_BYPASS_FIELDS = frozenset({
    "seed",
    "record_dir",
    "record_every",
    "record_resolution",
    "camera_id",
})


# Runtime knobs the MA user can pass to make_parallel (don't change the
# experiment identity / not part of the registered TaskSpec).
_MA_RUNTIME_KWARGS = frozenset({
    "obs_visibility",
    "reward_mode",
    "flatten_obs",
    "sustain_owner",  # currently unused — sustain_owner is fixed by AGENT_ASSIGNMENTS;
                      # accepted as a placeholder for Phase 2 override capability.
})


@dataclass(frozen=True)
class MATaskSpec:
    """MA env registration entry — pairs an MA id with an underlying SA env."""

    sa_env_id: str  # the registered SA env id whose dm_env chain we wrap
    morphology: str  # "ThreeHand" / "FourHand" / "FiveHand"


_ma_registry: Dict[str, MATaskSpec] = {}


def register_parallel(
    id: str,
    *,
    sa_env_id: str,
    morphology: str,
) -> None:
    """Register an MA env id pointing at an underlying SA env + morphology."""
    if morphology not in AGENT_ASSIGNMENTS:
        raise ValueError(
            f"unknown morphology {morphology!r}; "
            f"valid: {sorted(AGENT_ASSIGNMENTS.keys())}"
        )
    _ma_registry[id] = MATaskSpec(sa_env_id=sa_env_id, morphology=morphology)


def list_parallel_envs() -> Tuple[str, ...]:
    return tuple(sorted(_ma_registry.keys()))


def make_parallel(
    env_id: str,
    **kwargs,
) -> OmniPianoParallelEnv:
    """Build a PettingZoo ``ParallelEnv`` for an MA env id.

    Args:
        env_id: MA env id (must contain ``-MA-``). Looked up in the MA registry.
        **kwargs: a small set of runtime knobs (``seed``, ``record_*``, plus
            MA-specific: ``obs_visibility``, ``reward_mode``, ``flatten_obs``).
            Anything else raises ValueError.

    Returns:
        ``OmniPianoParallelEnv`` instance.
    """
    if env_id not in _ma_registry:
        # Helpful redirect if the user passed an SA env id.
        from OmniPiano.envs import registration as sa_reg
        if env_id in getattr(sa_reg, "_registry", {}):
            raise ValueError(
                f"env_id {env_id!r} is a single-agent env. Use "
                f"OmniPiano.make({env_id!r}) instead, or pick a multi-agent "
                f"env id (containing '-MA-'). "
                f"Available MA envs: {sorted(_ma_registry.keys())}"
            )
        raise ValueError(
            f"unknown MA env id: {env_id!r}. "
            f"Available: {sorted(_ma_registry.keys())}"
        )

    allowed = _RUNTIME_BYPASS_FIELDS | _MA_RUNTIME_KWARGS
    illegal = set(kwargs) - allowed
    if illegal:
        raise ValueError(
            f"make_parallel(**kwargs) only accepts {sorted(allowed)}. "
            f"Got illegal overrides: {sorted(illegal)}. To vary other "
            f"fields, register a new MA env id."
        )

    ma_spec = _ma_registry[env_id]
    morphology = ma_spec.morphology
    assignment = AGENT_ASSIGNMENTS[morphology]

    # MA runtime knobs (with defaults).
    obs_visibility = kwargs.pop("obs_visibility", "own_plus_boundary")
    reward_mode = kwargs.pop("reward_mode", "shared")
    flatten_obs = kwargs.pop("flatten_obs", False)
    _ = kwargs.pop("sustain_owner", None)  # placeholder (Phase 2 override)

    seed = kwargs.pop("seed", None)

    # ---- Build the env_builder closure (mirrors SA's _build_dm_env_chain) ----
    # We reuse SA registration's machinery by reaching in for the TaskSpec
    # (registered SA env spec → base_env_name + configs + hand_specs).
    from OmniPiano.envs.registration import _registry as _sa_registry
    if ma_spec.sa_env_id not in _sa_registry:
        raise ValueError(
            f"MA env {env_id!r} references unknown SA env id {ma_spec.sa_env_id!r}."
        )
    sa_spec = _sa_registry[ma_spec.sa_env_id]
    hand_specs = sa_spec.hand_specs
    if hand_specs is None:
        raise ValueError(
            f"MA env {env_id!r}: underlying SA env {ma_spec.sa_env_id!r} has no "
            f"hand_specs registered — MA requires explicit N-hand specs."
        )

    hand_key_ranges = {}
    for spec in hand_specs:
        if spec.key_range is None:
            raise ValueError(
                f"MA env {env_id!r}: hand_spec {spec.name!r} has no key_range. "
                f"MA only supports StaticPartition SA envs (Territorial series)."
            )
        hand_key_ranges[spec.name] = spec.key_range

    base_env_name = sa_spec.base_env_name
    robust_config = sa_spec.robust_config or RobustConfig()
    safety_config = sa_spec.safety_config or SafetyConfig()
    task_config = sa_spec.task_config or TaskVariantConfig()
    env_config = sa_spec.env_config or BenchmarkEnvConfig()

    # The env_builder closure — same as SA but stops before ConcatObservationWrapper.
    env_builder = _make_dm_env_chain_builder(
        base_env_name=base_env_name,
        env_config=env_config,
        task_config=task_config,
        robust_config=robust_config,
        hand_specs=hand_specs,
    )

    # Compute precise agent reach (cached).
    from OmniPiano.multiagent.assignment import compute_agent_reach
    agent_reaches = compute_agent_reach(assignment, hand_specs)

    return OmniPianoParallelEnv(
        env_builder=env_builder,
        assignment=assignment,
        hand_key_ranges=hand_key_ranges,
        agent_reaches=agent_reaches,
        seed=seed,
        obs_visibility=obs_visibility,
        reward_mode=reward_mode,
        flatten_obs=flatten_obs,
    )


# ===========================================================================
# dm_env chain builder (MA variant — skips ConcatObservationWrapper)
# ===========================================================================


def _make_dm_env_chain_builder(
    *,
    base_env_name: str,
    env_config: BenchmarkEnvConfig,
    task_config: TaskVariantConfig,
    robust_config: RobustConfig,
    hand_specs,
) -> Callable[[Optional[int]], dm_env.Environment]:
    """Return a closure that builds the MA-side dm_env chain (Dict obs preserved).

    The chain mirrors SA's ``_build_dm_env_chain`` EXACTLY except that
    ``ConcatObservationWrapper`` is omitted. All other wrappers
    (``EpisodeStatistics``, ``MidiEvaluation``, optional noise/OAR/FrameStack,
    ``CanonicalSpec``, ``SinglePrecision``) handle Dict obs natively via
    ``tree.map_structure`` and don't depend on Concat being upstream.
    """
    # Lazy import (heavy SA deps).
    from dm_env_wrappers import (
        CanonicalSpecWrapper,
        EpisodeStatisticsWrapper,
        FrameStackingWrapper,
        ObservationActionRewardWrapper,
        SinglePrecisionWrapper,
    )
    from OmniPiano.envs.dm_env_obs_noise import DmEnvObsNoiseWrapper
    from OmniPiano.envs.robopianist.wrappers import (
        MidiEvaluationWrapper,
    )
    from OmniPiano.tasks.omni_piano_task import OmniPianoTask
    from robopianist.suite import load_with_task as _suite_load_with_task  # noqa: F401
    # `robopianist.suite` here resolves to our vendored copy at
    # OmniPiano/envs/robopianist/suite/ (priority via sys.path bootstrap in
    # OmniPiano/__init__.py).

    # Snapshot env_config fields used by the chain.
    cfg = dataclasses.asdict(env_config)
    stretch = cfg.pop("stretch_factor")
    shift = cfg.pop("shift_factor")
    frame_stack = cfg.pop("frame_stack")
    clip = cfg.pop("clip")
    action_reward_observation = cfg.pop("action_reward_observation")
    # Drop record_* fields and any remaining we don't use (matches SA structure).
    for k in ("record_dir", "record_every", "record_resolution", "camera_id"):
        cfg.pop(k, None)

    # Anything else left in `cfg` is task-level (forwarded to OmniPianoTask).
    task_kwargs_base = dict(cfg)

    def _build(_seed: Optional[int]) -> dm_env.Environment:
        task_kwargs = {"task_config": task_config, **task_kwargs_base}
        task_kwargs["hand_specs"] = hand_specs
        env = _suite_load_with_task(
            environment_name=base_env_name,
            task_cls=OmniPianoTask,
            midi_file=None,
            seed=_seed,
            stretch=stretch,
            shift=shift,
            task_kwargs=task_kwargs,
        )
        env = EpisodeStatisticsWrapper(env, deque_size=1)
        env = MidiEvaluationWrapper(env, deque_size=1)

        if robust_config.obs_noise_std > 0:
            obs_noise_seed = (_seed + 31415) if _seed is not None else None
            env = DmEnvObsNoiseWrapper(
                env,
                noise_std=robust_config.obs_noise_std,
                seed=obs_noise_seed,
            )

        if action_reward_observation:
            env = ObservationActionRewardWrapper(env)

        # SKIP ConcatObservationWrapper — MA wants Dict obs.

        if frame_stack > 1:
            env = FrameStackingWrapper(
                env, num_frames=frame_stack, flatten=True
            )

        env = CanonicalSpecWrapper(env, clip=clip)
        env = SinglePrecisionWrapper(env)
        return env

    return _build
