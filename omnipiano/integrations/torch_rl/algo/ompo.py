"""Occupancy-Matching Policy Optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..env.evaluator import evaluate_policy, write_eval_summary
from ..env.observation import observation_indices
from ..log.checkpoint import save_checkpoint
from ..log.logging import ProgressLogger
from ..model.ompo.occupancy import (
    TransitionDiscriminator,
    TwinQNetwork,
    discriminator_loss,
    build_ompo_actor,
)
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class OMPOConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 24
    batch_size: int = 256
    buffer_size: int = 1_000_000
    local_buffer_size: int = 1_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    tau: float = 0.005
    updates_per_step: int = 3
    target_update_interval: int = 2
    discriminator_batch_size: int = 256
    discriminator_rounds: int = 10
    discriminator_updates: int = 20
    discriminator_observation: str = "dynamic"
    exponent: float = 1.5
    correction_coefficient: float = 0.001
    reward_max: float = 1.0
    reward_epsilon: float = 1e-6
    device: str = "cpu"


def make_config(args, device: str) -> OMPOConfig:
    return OMPOConfig(
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        local_buffer_size=args.ompo_local_buffer_size,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        tau=args.tau,
        updates_per_step=args.ompo_updates_per_step,
        discriminator_observation=args.ompo_discriminator_observation,
        exponent=args.ompo_exponent,
        correction_coefficient=args.ompo_correction_coeff,
        reward_max=args.ompo_reward_max,
        device=device,
    )


def _sample_actor(actor, observation):
    from tensordict import TensorDict
    from torchrl.objectives.sac import compute_log_prob

    tensordict = TensorDict({"observation": observation}, batch_size=observation.shape[:-1])
    distribution = actor.get_dist(tensordict)
    action = distribution.rsample()
    log_prob = compute_log_prob(distribution, action, "sample_log_prob").unsqueeze(-1)
    return action, log_prob


def _f_divergence(residual, exponent):
    return residual.abs().pow(exponent) / exponent


def _f_gradient(residual, exponent):
    return residual.clamp_min(0.0).pow(exponent - 1.0)


def _soft_update(target, source, tau):
    for target_parameter, parameter in zip(target.parameters(), source.parameters()):
        target_parameter.data.copy_(
            tau * parameter.data + (1.0 - tau) * target_parameter.data
        )


def _update_agent(
    actor,
    critic,
    target_critic,
    discriminator,
    batch,
    initial_batch,
    actor_optimizer,
    critic_optimizer,
    alpha_optimizer,
    log_alpha,
    config,
    update_step,
):
    observation = batch["observation"]
    action = batch["action"]
    next_observation = batch["next", "observation"]
    reward = (batch["next", "reward"] + config.reward_max).clamp_min(
        config.reward_epsilon
    )
    terminated = batch["next", "terminated"].to(reward.dtype)
    initial_observation = initial_batch["observation"]
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        occupancy_logit = discriminator(batch)
        corrected_reward = reward - config.correction_coefficient * occupancy_logit
        next_action, next_log_prob = _sample_actor(actor, next_observation)
        target_q1, target_q2 = target_critic(next_observation, next_action)
        target_min = torch.minimum(target_q1, target_q2)
        online_next_q1, online_next_q2 = critic(next_observation, next_action)
        mixed_q1 = 0.05 * online_next_q1 + 0.95 * target_min
        mixed_q2 = 0.05 * online_next_q2 + 0.95 * target_min
        target1 = corrected_reward + config.gamma * (1.0 - terminated) * (
            mixed_q1 - alpha * next_log_prob
        )
        target2 = corrected_reward + config.gamma * (1.0 - terminated) * (
            mixed_q2 - alpha * next_log_prob
        )
        nominal_target = reward + config.gamma * (1.0 - terminated) * (
            mixed_q1 - alpha * next_log_prob
        )

    q1, q2 = critic(observation, action)
    initial_action, _ = _sample_actor(actor, initial_observation)
    initial_q1, initial_q2 = critic(initial_observation, initial_action)
    residual1 = target1 - q1
    residual2 = target2 - q2
    critic_loss = (
        _f_divergence(residual1, config.exponent)
        + _f_divergence(residual2, config.exponent)
        + (1.0 - config.gamma)
        * config.correction_coefficient
        * (initial_q1 + initial_q2)
    ).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss_value = float("nan")
    alpha_loss_value = float("nan")
    if update_step % config.target_update_interval == 0:
        for parameter in critic.parameters():
            parameter.requires_grad = False
        next_action, next_log_prob = _sample_actor(actor, next_observation)
        target_q1, target_q2 = target_critic(next_observation, next_action)
        target_min = torch.minimum(target_q1, target_q2)
        online_next_q1, online_next_q2 = critic(next_observation, next_action)
        actor_target1 = corrected_reward + config.gamma * (1.0 - terminated) * (
            0.05 * online_next_q1 + 0.95 * target_min - alpha * next_log_prob
        )
        actor_target2 = corrected_reward + config.gamma * (1.0 - terminated) * (
            0.05 * online_next_q2 + 0.95 * target_min - alpha * next_log_prob
        )
        q1, q2 = critic(observation, action)
        initial_action, _ = _sample_actor(actor, initial_observation)
        initial_q1, initial_q2 = critic(initial_observation, initial_action)
        residual1 = actor_target1 - q1
        residual2 = actor_target2 - q2
        actor_loss = -0.5 * (
            _f_gradient(residual1, config.exponent).detach() * residual1
            + _f_gradient(residual2, config.exponent).detach() * residual2
            + (1.0 - config.gamma)
            * config.correction_coefficient
            * (initial_q1 + initial_q2)
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        for parameter in critic.parameters():
            parameter.requires_grad = True

        _, log_prob = _sample_actor(actor, observation)
        target_entropy = -action.shape[-1]
        alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        _soft_update(target_critic, critic, config.tau)
        actor_loss_value = float(actor_loss.detach())
        alpha_loss_value = float(alpha_loss.detach())

    return {
        "loss_critic": float(critic_loss.detach()),
        "loss_actor": actor_loss_value,
        "loss_alpha": alpha_loss_value,
        "alpha": float(log_alpha.exp().detach()),
        "corrected_target_delta": float((target1 - nominal_target).mean()),
    }


def train_ompo(
    env_id, seed, run_dir, protocol, config: OMPOConfig,
    train_env, periodic_eval_env, hidden_sizes=(256, 256), checkpoint_steps=(),
):
    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    observation_dim = observation_spec.shape[-1]
    action_dim = action_spec.shape[-1]
    actor = build_ompo_actor(
        observation_dim, action_dim, action_spec, hidden_sizes
    ).to(config.device)
    critic = TwinQNetwork(observation_dim, action_dim, hidden_sizes).to(config.device)
    target_critic = copy.deepcopy(critic).to(config.device)
    for parameter in target_critic.parameters():
        parameter.requires_grad = False
    indices = observation_indices(
        periodic_eval_env, config.discriminator_observation
    ).to(config.device)
    discriminator = TransitionDiscriminator(indices, action_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate
    )
    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.learning_rate)
    global_buffer = make_replay_buffer(config.buffer_size, config.batch_size)
    local_buffer = make_replay_buffer(config.local_buffer_size, config.discriminator_batch_size)
    initial_buffer = make_replay_buffer(config.buffer_size, config.batch_size)
    collector = make_collector(
        train_env, actor, config.n_envs, config.total_env_steps,
        config.device, config.learning_starts,
    )
    logger = ProgressLogger(run_dir / "progress.csv")
    model_spec = {
        "model_type": "ompo",
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
        "action_low": action_spec.space.low.tolist(),
        "action_high": action_spec.space.high.tolist(),
        "hidden_sizes": list(hidden_sizes),
    }
    env_steps = 0
    update_steps = 0
    next_eval = protocol.eval_freq_env_steps
    checkpoint_index = 0
    checkpoint_steps = sorted(checkpoint_steps)
    try:
        for batch in collector:
            collected = batch_env_steps(batch)
            env_steps += collected
            flat = batch.reshape(-1).cpu()
            flat["sample_step"] = torch.full(flat.batch_size, env_steps, dtype=torch.int64)
            global_buffer.extend(flat)
            local_buffer.extend(flat)
            initial_mask = flat["is_init"].squeeze(-1)
            if initial_mask.any():
                initial_buffer.extend(flat[initial_mask].select("observation"))

            discriminator_metrics = {
                "discriminator_loss": float("nan"),
                "discriminator_accuracy": float("nan"),
            }
            if (
                env_steps >= config.learning_starts
                and len(local_buffer) == config.local_buffer_size
            ):
                for _ in range(config.discriminator_rounds):
                    historical = make_replay_buffer(
                        config.local_buffer_size, config.discriminator_batch_size
                    )
                    historical.extend(global_buffer.sample(config.local_buffer_size))
                    for _ in range(config.discriminator_updates):
                        local_sample = local_buffer.sample().to(config.device)
                        global_sample = historical.sample().to(config.device)
                        loss, accuracy = discriminator_loss(
                            discriminator, local_sample, global_sample
                        )
                        discriminator_optimizer.zero_grad()
                        loss.backward()
                        discriminator_optimizer.step()
                        discriminator_metrics = {
                            "discriminator_loss": float(loss.detach()),
                            "discriminator_accuracy": float(accuracy.detach()),
                        }
                local_buffer.empty()

            agent_metrics = {
                "loss_critic": float("nan"), "loss_actor": float("nan"),
                "loss_alpha": float("nan"), "alpha": float("nan"),
                "corrected_target_delta": float("nan"),
            }
            if (
                env_steps >= config.learning_starts
                and len(global_buffer) >= config.batch_size
                and len(initial_buffer) > 0
            ):
                for _ in range(collected * config.updates_per_step):
                    sample = global_buffer.sample().to(config.device)
                    initial_sample = initial_buffer.sample().to(config.device)
                    agent_metrics = _update_agent(
                        actor, critic, target_critic, discriminator,
                        sample, initial_sample, actor_optimizer, critic_optimizer,
                        alpha_optimizer, log_alpha, config, update_steps,
                    )
                    update_steps += 1
                collector.update_policy_weights_()

            diagnostic_sample = global_buffer.sample(
                min(config.batch_size, len(global_buffer))
            ).to(config.device)
            with torch.no_grad():
                log_ratio = discriminator(diagnostic_sample)
                ratio = log_ratio.clamp(-20.0, 20.0).exp()
            ages = env_steps - diagnostic_sample["sample_step"].float()
            local_age = (
                env_steps - local_buffer.sample(min(len(local_buffer), config.batch_size))[
                    "sample_step"
                ].float()
            ).mean() if len(local_buffer) else torch.tensor(0.0)
            diagnostics = {
                "occupancy_ratio_mean": float(ratio.mean()),
                "occupancy_ratio_var": float(ratio.var(unbiased=False)),
                "occupancy_ratio_p10": float(ratio.quantile(0.1)),
                "occupancy_ratio_p90": float(ratio.quantile(0.9)),
                "global_buffer_age": float(ages.mean()),
                "local_buffer_age": float(local_age),
            }
            logger.write(
                {"env_steps": env_steps, "global_buffer_size": len(global_buffer),
                 "local_buffer_size": len(local_buffer), **agent_metrics,
                 **discriminator_metrics, **diagnostics}
            )
            if env_steps >= next_eval:
                adapter = TorchRLPolicyAdapter(actor, model_spec, config.device)
                result = evaluate_policy(
                    adapter, env_id, periodic_eval_env, seed + 10_000,
                    protocol.num_eval_eps,
                )
                result["training_step"] = env_steps
                write_eval_summary(result, run_dir / "periodic_eval" / f"step_{env_steps}.json")
                while next_eval <= env_steps:
                    next_eval += protocol.eval_freq_env_steps
            while checkpoint_index < len(checkpoint_steps) and env_steps >= checkpoint_steps[checkpoint_index]:
                target = checkpoint_steps[checkpoint_index]
                save_checkpoint(
                    run_dir / "checkpoints" / f"checkpoint_{target}",
                    actor=actor, model_spec=model_spec,
                )
                checkpoint_index += 1
            if env_steps >= config.total_env_steps:
                break
    finally:
        collector.shutdown()
        logger.close()

    final_dir = save_checkpoint(run_dir / "final_model", actor=actor, model_spec=model_spec)
    return TorchRLPolicyAdapter.load(final_dir, config.device), None
