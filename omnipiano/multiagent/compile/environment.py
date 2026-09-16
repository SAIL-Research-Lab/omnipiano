"""Runtime bridge: one resolved task snapshot -> one PettingZoo environment."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from omnipiano.multiagent.compile.schema import ResolvedAgent, ResolvedHand, ResolvedTask


def _plain(value):
    return (
        dataclasses.asdict(value)
        if dataclasses.is_dataclass(value)
        else dict(value)
    )


def _serialize_hand_spec(spec: Any) -> Dict[str, Any]:
    data = dataclasses.asdict(spec)
    data["side"] = spec.side.name
    return data


def resolve_registered_task(env_id: str) -> ResolvedTask:
    """Translate one historical MA registry entry into the common task IR.

    The returned snapshot contains every trajectory-affecting SA config and
    the already-expanded per-agent wrist clamps.  It can therefore cross a
    Ray process boundary without relying on worker-local registry state.
    """
    from omnipiano.configs import (
        BenchmarkEnvConfig,
        RobustConfig,
        TaskVariantConfig,
    )
    from omnipiano.envs import registration as sa_reg
    from omnipiano.multiagent.compile.env_runtime.topology import (
        AGENT_ASSIGNMENTS,
        compute_agent_territory,
        compute_boundary_hands,
    )
    from omnipiano.multiagent.compile.presets import SONGS
    from omnipiano.multiagent.registration import _ma_registry

    if env_id not in _ma_registry:
        if env_id in getattr(sa_reg, "_registry", {}):
            raise ValueError(
                f"env_id {env_id!r} is a single-agent env. Use "
                f"omnipiano.make({env_id!r}) instead, or choose a registered "
                "multi-agent env id."
            )
        raise ValueError(
            f"unknown MA env id: {env_id!r}. Available: "
            f"{sorted(_ma_registry.keys())}"
        )

    ma_spec = _ma_registry[env_id]
    if ma_spec.sa_env_id not in sa_reg._registry:
        raise ValueError(
            f"MA env {env_id!r} references unknown SA env id "
            f"{ma_spec.sa_env_id!r}."
        )
    sa_spec = sa_reg._registry[ma_spec.sa_env_id]
    assignment = AGENT_ASSIGNMENTS[ma_spec.morphology]
    robust_config = sa_spec.robust_config or RobustConfig()
    task_config = sa_spec.task_config or TaskVariantConfig()
    env_config = sa_spec.env_config or BenchmarkEnvConfig()

    # The MA dm_env chain does not contain the SA gym-level safety/action-
    # robustness wrappers.  Reject unsupported registrations before creating
    # an apparently valid but scientifically different environment.
    if sa_spec.safety_config is not None and sa_spec.safety_config.constraints:
        raise ValueError(
            f"MA env {env_id!r}: underlying SA env {ma_spec.sa_env_id!r} has "
            "active safety constraints, but the MA chain has no SafetyWrapper."
        )
    if sa_spec.robust_config is not None:
        for channel in ("action", "reward"):
            if sa_spec.robust_config.is_channel_active(channel):
                raise ValueError(
                    f"MA env {env_id!r}: underlying SA env "
                    f"{ma_spec.sa_env_id!r} has an active {channel!r} robust "
                    "channel, which the MA chain cannot apply."
                )
    if robust_config.is_channel_active("obs"):
        sa_mode = getattr(sa_spec, "mode", None)
        if sa_mode is not None and str(sa_mode) != "train":
            raise ValueError(
                f"MA env {env_id!r}: mode={sa_mode!r} requires SA eval-noise "
                "scaling, which the MA chain does not implement."
            )
    if env_config.frame_stack > 1:
        raise NotImplementedError(
            f"MA env {env_id!r}: frame_stack={env_config.frame_stack} is not "
            "supported because the MA observation remains a Dict."
        )
    if sa_spec.hand_specs is None:
        raise ValueError(
            f"MA env {env_id!r}: underlying SA env {ma_spec.sa_env_id!r} has "
            "no explicit N-hand specs."
        )

    hand_specs = tuple(sa_spec.hand_specs)
    hand_key_ranges: Dict[str, Tuple[int, int]] = {}
    for spec in hand_specs:
        if spec.key_range is None:
            raise ValueError(
                f"MA env {env_id!r}: hand {spec.name!r} has no key_range; "
                "registered MA tasks require a static partition."
            )
        hand_key_ranges[spec.name] = tuple(spec.key_range)
    territories = compute_agent_territory(assignment, hand_key_ranges)
    clamped_specs = tuple(
        dataclasses.replace(
            spec, key_range=territories[assignment.agent_of(spec.name)]
        )
        for spec in hand_specs
    )
    hand_index = {spec.name: index for index, spec in enumerate(hand_specs)}
    boundary_hands = compute_boundary_hands(assignment, hand_key_ranges)
    agents = tuple(
        ResolvedAgent(
            name=agent.name,
            hand_ids=tuple(hand_index[name] for name in agent.hand_names),
            action_key_range=territories[agent.name],
            observation_key_range=None,
            visible_teammate_hands=tuple(
                sorted(
                    boundary_hands[agent.name].values(),
                    key=hand_index.__getitem__,
                )
            ),
            is_sustain_owner=agent.is_sustain_owner,
        )
        for agent in assignment.agents
    )
    song = next(
        (name for name, base_env_name in SONGS.items()
         if base_env_name == sa_spec.base_env_name),
        sa_spec.base_env_name,
    )
    return ResolvedTask(
        name=env_id,
        song=song,
        base_env_name=sa_spec.base_env_name,
        layout=ma_spec.morphology,
        assignment_mode="registered",
        hands=tuple(
            ResolvedHand(index, spec.name, hand_key_ranges[spec.name])
            for index, spec in enumerate(hand_specs)
        ),
        agents=agents,
        legacy_env_id=env_id,
        hand_specs=tuple(_serialize_hand_spec(spec) for spec in clamped_specs),
        env_config=_plain(env_config),
        task_config=_plain(task_config),
        robust_config=_plain(robust_config),
    )


def prepare_task(task: ResolvedTask) -> ResolvedTask:
    """Attach complete hand/environment snapshots before Ray workers start."""
    if task.hand_specs:
        return task
    from omnipiano.configs import (
        BenchmarkEnvConfig,
        RobustConfig,
        TaskVariantConfig,
    )
    from omnipiano.tasks.hand_spec import default_partitioned_hand_specs

    specs = default_partitioned_hand_specs(len(task.hands))
    if tuple(s.name for s in specs) != tuple(h.name for h in task.hands):
        raise RuntimeError("compiled layout and physical hand preset disagree")
    owner = {h: a for a in task.agents for h in a.hand_ids}
    action_ranges = {task.hands[h].name: owner[h].action_key_range for h in owner}
    serialized = []
    for spec in specs:
        updated = dataclasses.replace(spec, key_range=action_ranges[spec.name])
        serialized.append(_serialize_hand_spec(updated))
    return dataclasses.replace(
        task,
        hand_specs=tuple(serialized),
        env_config=_plain(BenchmarkEnvConfig(disable_fingering_reward=True)),
        task_config=_plain(TaskVariantConfig()),
        robust_config=_plain(RobustConfig()),
    )


def make_parallel_from_task(
    task: ResolvedTask | Mapping[str, Any],
    *,
    seed: Optional[int] = None,
    record_dir: Optional[str] = None,
    record_every: int = 1,
    record_resolution: Tuple[int, int] = (480, 640),
    camera_id: str = "piano/back",
    flatten_obs: bool = False,
    include_global_state: bool = False,
    inter_agent_collision_penalty_coef: float = 0.0,
    obs_visibility: str = "own_plus_boundary",
    reward_mode: str = "shared",
    sustain_owner: Optional[str] = None,
):
    """Build a custom task without mutating either process-local registry."""
    if not isinstance(task, ResolvedTask):
        task = ResolvedTask.from_dict(task)
    task = prepare_task(task)

    from robopianist.models.hands import HandSide
    from omnipiano.configs import (
        BenchmarkEnvConfig,
        RobustConfig,
        TaskVariantConfig,
    )
    from omnipiano.multiagent.compile.env_runtime.topology import (
        AgentDef,
        MorphologyAssignment,
        compute_agent_reach,
    )
    from omnipiano.multiagent.compile.env_runtime.parallel_env import (
        OmniPianoParallelEnv,
    )
    from omnipiano.tasks.hand_spec import HandSpec

    hand_specs = tuple(HandSpec(
        name=h["name"], side=HandSide[h["side"]],
        position=tuple(h["position"]), quaternion=tuple(h["quaternion"]),
        attachment_yaw=h["attachment_yaw"], forearm_dofs=tuple(h["forearm_dofs"]),
        reduced_action_space=h["reduced_action_space"], group=h["group"],
        y_range=tuple(h["y_range"]) if h["y_range"] is not None else None,
        key_range=tuple(h["key_range"]) if h["key_range"] is not None else None,
    ) for h in task.hand_specs)
    assignment = MorphologyAssignment(
        morphology=task.layout,
        agent_setup=f"Custom{len(task.agents)}Agent",
        agents=tuple(AgentDef(
            a.name, tuple(task.hands[h].name for h in a.hand_ids), a.is_sustain_owner
        ) for a in task.agents),
    )
    env_config = BenchmarkEnvConfig(**dict(task.env_config or {}))
    task_config = TaskVariantConfig(**dict(task.task_config or {}))
    robust_config = RobustConfig(**dict(task.robust_config or {}))
    env_builder = _make_dm_env_chain_builder(
        base_env_name=task.base_env_name, env_config=env_config,
        task_config=task_config, robust_config=robust_config,
        hand_specs=hand_specs, record_dir=record_dir, record_every=record_every,
        record_resolution=record_resolution, camera_id=camera_id,
    )
    reaches = compute_agent_reach(
        assignment, hand_specs,
        control_timestep=env_config.control_timestep,
        gravity_compensation=env_config.gravity_compensation,
        disable_hand_collisions=env_config.disable_hand_collisions,
    )
    observation_ranges = {
        a.name: (a.observation_key_range
                 if a.observation_key_range is not None else reaches[a.name])
        for a in task.agents
    }
    return OmniPianoParallelEnv(
        env_builder=env_builder, assignment=assignment,
        hand_key_ranges={h.name: h.bucket_key_range for h in task.hands},
        agent_reaches=reaches, seed=seed, flatten_obs=flatten_obs,
        include_global_state=include_global_state,
        inter_agent_collision_penalty_coef=inter_agent_collision_penalty_coef,
        obs_visibility=obs_visibility, reward_mode=reward_mode,
        sustain_owner=sustain_owner,
        observation_key_ranges=observation_ranges,
        visible_teammate_hands={
            a.name: a.visible_teammate_hands for a in task.agents
        },
    )


def _make_dm_env_chain_builder(
    *,
    base_env_name: str,
    env_config: Any,
    task_config: Any,
    robust_config: Any,
    hand_specs: Any,
    record_dir: Optional[str] = None,
    record_every: int = 1,
    record_resolution: Tuple[int, int] = (480, 640),
    camera_id: str = "piano/back",
) -> Callable[[Optional[int]], Any]:
    """Build the MA dm_env chain while preserving the observation Dict."""
    from dm_env_wrappers import (
        CanonicalSpecWrapper,
        EpisodeStatisticsWrapper,
        ObservationActionRewardWrapper,
        SinglePrecisionWrapper,
    )
    from omnipiano.envs.dm_env_obs_noise import (
        DmEnvObsNoiseWrapper,
        OBS_NOISE_SEED_OFFSET,
    )
    from omnipiano.envs.robopianist.wrappers import MidiEvaluationWrapper
    from omnipiano.tasks.omni_piano_task import OmniPianoTask
    from robopianist.suite import load_with_task as suite_load_with_task

    config = dataclasses.asdict(env_config)
    stretch = config.pop("stretch_factor")
    shift = config.pop("shift_factor")
    frame_stack = config.pop("frame_stack")
    clip = config.pop("clip")
    action_reward_observation = config.pop("action_reward_observation")
    for key in ("record_dir", "record_every", "record_resolution", "camera_id"):
        config.pop(key, None)
    task_kwargs_base = dict(config)

    def _build(env_seed: Optional[int]):
        task_kwargs = {"task_config": task_config, **task_kwargs_base}
        task_kwargs["hand_specs"] = hand_specs
        env = suite_load_with_task(
            environment_name=base_env_name,
            task_cls=OmniPianoTask,
            midi_file=None,
            seed=env_seed,
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
                height=record_resolution[0],
                width=record_resolution[1],
            )
        env = MidiEvaluationWrapper(env, deque_size=1)
        if robust_config.is_channel_active("obs"):
            obs_noise_seed = (
                env_seed + OBS_NOISE_SEED_OFFSET
                if env_seed is not None else None
            )
            env = DmEnvObsNoiseWrapper(
                env, robust_config=robust_config, seed=obs_noise_seed
            )
        if action_reward_observation:
            env = ObservationActionRewardWrapper(env)
        if frame_stack > 1:
            raise NotImplementedError(
                "frame stacking is unsupported for the MA Dict observation"
            )
        env = CanonicalSpecWrapper(env, clip=clip)
        return SinglePrecisionWrapper(env)

    return _build


__all__ = [
    "make_parallel_from_task",
    "prepare_task",
    "resolve_registered_task",
]
