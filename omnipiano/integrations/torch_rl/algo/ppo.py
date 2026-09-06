"""Phase-0 TorchRL PPO baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..env.evaluator import evaluate_policy, write_eval_summary
from ..log.checkpoint import save_checkpoint
from ..log.logging import ProgressLogger, TrainingEpisodeTracker
from ..model.networks import build_ppo_modules
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class PPOConfig:
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
    critic_coeff: float = 0.5
    device: str = "cpu"


def make_config(args, device: str) -> PPOConfig:
    return PPOConfig(
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        device=device,
    )


def train_ppo(
    env_id, seed, run_dir, protocol, config: PPOConfig, train_env, periodic_eval_env,
    hidden_sizes=(256, 256), checkpoint_steps=(),
):
    from tensordict.nn import TensorDictSequential
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    obs_dim = observation_spec.shape[-1]
    action_dim = action_spec.shape[-1]
    actor, critic = build_ppo_modules(obs_dim, action_dim, action_spec, hidden_sizes)
    actor.to(config.device)
    critic.to(config.device)
    model_spec = {
        "observation_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": action_spec.space.low.tolist(),
        "action_high": action_spec.space.high.tolist(),
        "hidden_sizes": list(hidden_sizes),
    }
    rollout_size = config.n_envs * config.n_steps
    collection_policy = TensorDictSequential(actor, critic)
    collector = make_collector(
        train_env, collection_policy, rollout_size, config.total_env_steps, config.device
    )
    advantage = GAE(
        gamma=config.gamma, lmbda=config.gae_lambda, value_network=critic,
        average_gae=True,
    )
    loss_module = ClipPPOLoss(
        actor, critic, clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_coeff, critic_coeff=config.critic_coeff,
    )
    loss_module.to(config.device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.learning_rate)
    minibatches = make_replay_buffer(rollout_size, config.batch_size)
    logger = ProgressLogger(run_dir / "progress.csv", sb3_family="ppo")
    episode_tracker = TrainingEpisodeTracker()
    update_steps = 0
    env_steps = 0
    next_eval = protocol.eval_freq_env_steps
    checkpoint_index = 0
    checkpoint_steps = sorted(checkpoint_steps)
    try:
        for rollout in collector:
            rollout = rollout.to(config.device)
            env_steps += batch_env_steps(rollout)
            rollout_stats = episode_tracker.update(rollout)
            # GAE evaluates current and next observations together; remove the
            # collector-side value so both sides have the same input keys.
            rollout.del_("state_value")
            with torch.no_grad():
                advantage(rollout)
            flat = rollout.reshape(-1)
            minibatches.empty()
            minibatches.extend(flat)
            latest = {}
            updates_per_epoch = max(len(minibatches) // config.batch_size, 1)
            for _ in range(config.n_epochs * updates_per_epoch):
                losses = loss_module(minibatches.sample())
                total_loss = sum(
                    value for key, value in losses.items() if str(key).startswith("loss_")
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                latest = {key: float(value.detach().mean()) for key, value in losses.items()}
                latest["loss"] = float(total_loss.detach().mean())
            update_steps += config.n_epochs
            if "state_value" in rollout.keys() and "value_target" in rollout.keys():
                predicted = rollout["state_value"].detach()
                target = rollout["value_target"].detach()
                target_variance = target.var(unbiased=False)
                latest["explained_variance"] = float(
                    1.0 - (target - predicted).var(unbiased=False)
                    / target_variance.clamp_min(1e-8)
                )
            if "scale" in rollout.keys():
                latest["std"] = float(rollout["scale"].detach().mean())
            collector.update_policy_weights_()
            eval_stats = {
                "eval/mean_reward": float("nan"),
                "eval/mean_f1": float("nan"),
                "eval/mean_ep_length": float("nan"),
            }
            did_eval = False
            if env_steps >= next_eval:
                adapter = TorchRLPolicyAdapter(actor, model_spec, config.device)
                result = evaluate_policy(
                    adapter, env_id, periodic_eval_env, seed + 10_000,
                    protocol.num_eval_eps,
                )
                result["training_step"] = env_steps
                write_eval_summary(result, run_dir / "periodic_eval" / f"step_{env_steps}.json")
                eval_stats.update({
                    "eval/mean_reward": result["summary"]["ep_return"],
                    "eval/mean_f1": result["summary"]["ep_f1"],
                    "eval/mean_ep_length": result["summary"]["ep_length"],
                })
                did_eval = True
                while next_eval <= env_steps:
                    next_eval += protocol.eval_freq_env_steps
            logger.write({
                "env_steps": env_steps,
                **rollout_stats,
                "learning_rate": config.learning_rate,
                "n_updates": update_steps,
                "clip_range": config.clip_epsilon,
                **latest,
                **eval_stats,
            }, force=did_eval)
            while (
                checkpoint_index < len(checkpoint_steps)
                and env_steps >= checkpoint_steps[checkpoint_index]
            ):
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

    final_dir = save_checkpoint(
        run_dir / "final_model", actor=actor, model_spec=model_spec,
    )
    return TorchRLPolicyAdapter.load(final_dir, config.device), None
