"""Evidential Proximal Policy Optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..env.evaluator import evaluate_policy, write_eval_summary
from ..log.checkpoint import save_checkpoint
from ..log.logging import ProgressLogger
from ..model.eppo.evidential import (
    build_evidential_critic,
    nig_loss,
    uncertainty,
    value_variance,
)
from ..model.networks import build_actor
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class EPPOConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 16
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.0
    kappa: float = 0.01
    mode: str = "cor"
    evidential_regularization: float = 0.01
    evidential_eps: float = 1e-6
    device: str = "cpu"


def make_config(args, device: str) -> EPPOConfig:
    return EPPOConfig(
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        mode=args.eppo_mode,
        kappa=args.kappa,
        evidential_regularization=args.evidential_reg,
        device=device,
    )


def _advantages(rollout, critic, config: EPPOConfig):
    """Compute EPPO mean/UCB advantages following Algorithm 1."""
    with torch.no_grad():
        critic(rollout)
        critic(rollout["next"])
        values = rollout["state_value"]
        flat_values = values.reshape(-1, values.shape[-1])
        flat_next_values = rollout["next", "state_value"].reshape_as(flat_values)
        flat_variances = value_variance(rollout).reshape_as(flat_values)
        flat_next_variances = value_variance(rollout["next"]).reshape_as(flat_values)
        flat_rewards = rollout["next", "reward"].reshape_as(flat_values)
        flat_done = rollout["next", "done"].reshape_as(flat_values).to(values.dtype)
        traj_ids = rollout["collector", "traj_ids"].reshape(-1)
        mask = 1.0 - flat_done
        delta = (
            flat_rewards
            + config.gamma * flat_next_values * mask
            - flat_values
        )

        layout = _trajectory_layout(traj_ids)
        advantage = _segmented_reverse_scan(
            delta,
            config.gamma * config.gae_lambda * mask,
            layout,
        )
        advantage_var = torch.zeros_like(flat_values)
        if config.mode != "mean":
            gae_squared = (config.gamma * config.gae_lambda) ** 2
            variance_acc = _segmented_reverse_scan(
                gae_squared * flat_next_variances,
                gae_squared * mask,
                layout,
            )
            current_scale = (
                1.0 if config.mode == "cor"
                else (1.0 - config.gae_lambda) / (1.0 + config.gae_lambda)
            )
            advantage_var = current_scale * flat_variances + (
                (1.0 - config.gae_lambda) / config.gae_lambda
            ) ** 2 * variance_acc

        ucb_bonus = config.kappa * advantage_var.clamp_min(0.0).sqrt()
        policy_advantage = advantage if config.mode == "mean" else advantage + ucb_bonus
        rollout["value_target"] = (advantage + flat_values).reshape_as(values)
        rollout["advantage"] = ((
            policy_advantage - policy_advantage.mean()
        ) / policy_advantage.std(unbiased=False).clamp_min(1e-8)).reshape_as(values)
        aleatoric, epistemic = uncertainty(rollout)
    return {
        "aleatoric_uncertainty": float(aleatoric.mean()),
        "epistemic_uncertainty": float(epistemic.mean()),
        "ucb_bonus": float(ucb_bonus.mean()),
    }


def _trajectory_layout(traj_ids):
    """Build vectorized row/column indices for interleaved trajectories."""
    order = torch.argsort(traj_ids, stable=True)
    sorted_ids = traj_ids[order]
    _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    starts = counts.cumsum(0) - counts
    rows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=traj_ids.device), counts
    )
    positions = torch.arange(
        traj_ids.numel(), device=traj_ids.device
    ) - torch.repeat_interleave(starts, counts)
    return order, rows, positions, counts.numel(), int(counts.max())


def _segmented_reverse_scan(source, continuation, layout):
    """Compute ``x[t] = source[t] + continuation[t] * x[t+1]`` per trajectory."""
    order, rows, positions, num_trajectories, max_length = layout
    tail_shape = source.shape[1:]
    padded_shape = (num_trajectories, max_length, *tail_shape)
    padded_source = source.new_zeros(padded_shape)
    padded_continuation = continuation.new_zeros(padded_shape)
    padded_source[rows, positions] = source[order]
    padded_continuation[rows, positions] = continuation[order]

    padded_result = source.new_zeros(padded_shape)
    accumulator = source.new_zeros((num_trajectories, *tail_shape))
    for time_index in range(max_length - 1, -1, -1):
        accumulator = (
            padded_source[:, time_index]
            + padded_continuation[:, time_index] * accumulator
        )
        padded_result[:, time_index] = accumulator

    result = torch.empty_like(source)
    result[order] = padded_result[rows, positions]
    return result


def train_eppo(
    env_id, seed, run_dir, protocol, config: EPPOConfig, train_env, periodic_eval_env,
    hidden_sizes=(256, 256), checkpoint_steps=(),
):
    from tensordict.nn import TensorDictSequential
    from torchrl.objectives import ClipPPOLoss

    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    obs_dim = observation_spec.shape[-1]
    action_dim = action_spec.shape[-1]
    actor = build_actor(obs_dim, action_dim, action_spec, hidden_sizes).to(config.device)
    critic = build_evidential_critic(obs_dim, hidden_sizes, config.evidential_eps).to(config.device)
    model_spec = {
        "observation_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": action_spec.space.low.tolist(),
        "action_high": action_spec.space.high.tolist(),
        "hidden_sizes": list(hidden_sizes),
    }
    rollout_size = config.n_envs * config.n_steps
    collector = make_collector(
        train_env, TensorDictSequential(actor, critic), rollout_size,
        config.total_env_steps, config.device,
    )
    ppo_loss = ClipPPOLoss(
        actor, critic, clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_coeff, critic_coeff=0.0,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    minibatches = make_replay_buffer(rollout_size, config.batch_size)
    logger = ProgressLogger(run_dir / "progress.csv")
    env_steps = 0
    next_eval = protocol.eval_freq_env_steps
    checkpoint_index = 0
    checkpoint_steps = sorted(checkpoint_steps)
    try:
        for rollout in collector:
            rollout = rollout.to(config.device)
            env_steps += batch_env_steps(rollout)
            rollout.del_("state_value")
            diagnostics = _advantages(rollout, critic, config)
            minibatches.empty()
            minibatches.extend(rollout.reshape(-1))
            latest = {}
            updates_per_epoch = max(len(minibatches) // config.batch_size, 1)
            for _ in range(config.n_epochs * updates_per_epoch):
                sample = minibatches.sample()
                ppo_terms = ppo_loss(sample)
                actor_loss = ppo_terms["loss_objective"] + ppo_terms["loss_entropy"]
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                critic_loss, value_nll, regularizer = nig_loss(
                    sample, critic, config.evidential_regularization
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()
                latest = {
                    "loss_actor": float(actor_loss.detach()),
                    "loss_critic": float(critic_loss.detach()),
                    "value_nll": float(value_nll.detach()),
                    "evidential_regularizer": float(regularizer.detach()),
                }
            logger.write({"env_steps": env_steps, **latest, **diagnostics})
            collector.update_policy_weights_()
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
                    actor=actor,
                    model_spec=model_spec,
                )
                checkpoint_index += 1
            if env_steps >= config.total_env_steps:
                break
    finally:
        collector.shutdown()
        logger.close()

    final_dir = save_checkpoint(run_dir / "final_model", actor=actor, model_spec=model_spec)
    return TorchRLPolicyAdapter.load(final_dir, config.device), None
