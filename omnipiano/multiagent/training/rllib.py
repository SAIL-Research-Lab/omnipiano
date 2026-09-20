"""RLlib-specific environment and algorithm assembly for MARL training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from omnipiano.multiagent.algos import AlgoSpec
from omnipiano.multiagent.training.runtime import (
    LEGACY_RLLIB_ENV_NAME,
    RLLIB_ENV_NAME,
    wrap_parallel_env_for_rllib,
)

RLLIB_TARGET_VERSION = "2.55.1"


def make_env_for_rllib(env_config: Mapping[str, Any]) -> Any:
    """RLlib env creator. ``include_global_state`` arrives via env_config.

    Each env runner gets a DISTINCT env seed. ``EnvContext`` carries
    worker_index / vector_index; a plain dict (evaluation path) does not, so
    both fall back to 0 and the driver-side probe stays reproducible.
    """
    from omnipiano.multiagent.compile.environment import (
        make_parallel_from_task,
        resolve_registered_task,
    )

    base_seed = int(env_config.get("seed", 0))
    worker_index = int(getattr(env_config, "worker_index", 0) or 0)
    vector_index = int(getattr(env_config, "vector_index", 0) or 0)

    task = env_config.get("task")
    if task is None:
        task = resolve_registered_task(str(env_config["env_id"])).to_dict()
    parallel_env = make_parallel_from_task(
        task,
        seed=base_seed + 1_000 * worker_index + vector_index,
        flatten_obs=True,
        include_global_state=bool(env_config.get("include_global_state", False)),
        inter_agent_collision_penalty_coef=float(
            env_config.get("inter_agent_collision_penalty_coef", 0.0)
        ),
    )
    # RLlib permits only ``__common__`` as a non-agent info key; the native
    # PettingZoo env emits ``_global_``. Translate at this adapter boundary.
    return wrap_parallel_env_for_rllib(parallel_env)


def probe_agent_spaces(
    env_id: str, seed: int, include_global_state: bool,
    task: Optional[Mapping[str, Any]] = None,
) -> Tuple[Sequence[str], Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Tuple[int, int]]]]:
    """Return agents, obs/action spaces, and the flat actor/critic slice layout."""
    from omnipiano.multiagent.compile.environment import (
        make_parallel_from_task,
        resolve_registered_task,
    )

    if task is None:
        task = resolve_registered_task(env_id).to_dict()
    env = make_parallel_from_task(
        task, seed=seed, flatten_obs=True,
        include_global_state=include_global_state,
    )
    try:
        agents = list(env.possible_agents)
        obs = {a: env.observation_space(a) for a in agents}
        act = {a: env.action_space(a) for a in agents}
        layouts: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for agent in agents:
            layout = env.obs_layout(agent)
            if "own" in layout:
                own, glob = layout["own"], layout.get("global_state", (0, 0))
            else:
                # No global state: the whole vector is the agent's own obs.
                own, glob = (0, int(np.prod(obs[agent].shape))), (0, 0)
            layouts[agent] = {"own": own, "global_state": glob}
        return agents, obs, act, layouts
    finally:
        env.close()


def register_rllib_envs() -> None:
    """Register both current and legacy RLlib environment names."""
    from ray.tune.registry import register_env

    register_env(RLLIB_ENV_NAME, make_env_for_rllib)
    register_env(LEGACY_RLLIB_ENV_NAME, make_env_for_rllib)


def build_ppo_config(
    args: Any,
    spec: AlgoSpec,
    agents: Sequence[str],
    obs_spaces: Mapping[str, Any],
    act_spaces: Mapping[str, Any],
    layouts: Mapping[str, Mapping[str, Tuple[int, int]]],
) -> Tuple[Any, Optional[type]]:
    """Build the RLlib PPO configuration and resolve its learner class."""
    if (
        not spec.is_on_policy
        or spec.rl_module not in ("ctde", "rllib_default")
    ):
        raise NotImplementedError(
            f"{spec.name}: build_ppo_config only supports the current "
            "per-agent PPO path. Joint/autoregressive and off-policy "
            "algorithms must use their dedicated training backend."
        )
    from ray.rllib.algorithms.ppo import PPOConfig

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .environment(RLLIB_ENV_NAME, env_config={
            "env_id": args.env_id, "seed": args.seed,
            "task": getattr(args, "_resolved_task", None),
            "include_global_state": bool(spec.needs_global_state),
            "inter_agent_collision_penalty_coef": float(
                args.inter_agent_collision_penalty_coef
            ),
        })
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers,
                     num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                     sample_timeout_s=args.sample_timeout_s)
        .learners(num_learners=args.num_learners,
                  num_gpus_per_learner=args.num_gpus_per_learner)
        .multi_agent(
            policies={a: (None, obs_spaces[a], act_spaces[a], {}) for a in agents},
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
            count_steps_by="env_steps",
        )
        .training(
            gamma=args.gamma, lr=args.lr,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size, num_epochs=args.num_epochs,
            lambda_=args.gae_lambda, clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param, vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff, use_kl_loss=args.use_kl_loss,
            grad_clip=args.grad_clip, grad_clip_by=args.grad_clip_by,
        )
        .debugging(seed=args.seed)
    )
    if spec.training_overrides:
        config = config.training(**dict(spec.training_overrides))

    if spec.num_agents_override is not None:
        raise NotImplementedError(
            f"--algo {spec.name} declares num_agents_override="
            f"{spec.num_agents_override}, but the environment/evaluation path "
            "does not thread it. Refusing a silent no-op."
        )

    if spec.rl_module == "ctde":
        from omnipiano.multiagent.algos.ppo_module import (
            build_ctde_module_spec, build_multi_module_spec,
        )
        from omnipiano.multiagent.algos.ppo_learner import OmniPianoPPOTorchLearner

        learner_cls = spec.resolve_learner_class() or OmniPianoPPOTorchLearner
        if not issubclass(learner_cls, OmniPianoPPOTorchLearner):
            raise TypeError(
                f"--algo {spec.name}: learner_class {learner_cls.__name__} must "
                "subclass OmniPianoPPOTorchLearner for rl_module='ctde'"
            )
        config = (
            config.learners(
                learner_class=learner_cls,
                learner_config_dict={
                    "critic_lr": float(args.critic_lr),
                    "adam_epsilon": float(args.adam_epsilon),
                },
            )
            .rl_module(rl_module_spec=build_multi_module_spec({
                agent: build_ctde_module_spec(
                    observation_space=obs_spaces[agent],
                    action_space=act_spaces[agent],
                    own_slice=layouts[agent]["own"],
                    global_state_slice=layouts[agent]["global_state"],
                    critic_input=spec.critic_input,
                    hidden_sizes=args.hidden_sizes_parsed,
                    activation=args.activation,
                    hidden_orthogonal_gain=args.hidden_orthogonal_gain,
                    policy_output_gain=args.policy_output_gain,
                    value_output_gain=args.value_output_gain,
                    initial_log_std=args.initial_log_std,
                    log_std_min=args.log_std_min,
                    log_std_max=args.log_std_max,
                    input_layer_norm=args.input_layer_norm,
                    value_norm=args.value_norm,
                    value_norm_beta=args.value_norm_beta,
                    value_norm_epsilon=args.value_norm_epsilon,
                    value_norm_variance_floor=args.value_norm_variance_floor,
                )
                for agent in agents
            }))
        )
    else:
        learner_cls = spec.resolve_learner_class()
        if learner_cls is not None:
            config = config.learners(learner_class=learner_cls)

    return config, learner_cls
