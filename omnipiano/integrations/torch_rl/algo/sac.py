"""Phase-0 TorchRL SAC baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..env.evaluator import evaluate_policy, write_eval_summary
from ..log.checkpoint import save_checkpoint
from ..log.logging import ProgressLogger
from ..model.networks import build_sac_modules
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class SACConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 24
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    tau: float = 0.005
    utd: int = 1
    device: str = "cpu"


def make_config(args, device: str) -> SACConfig:
    return SACConfig(
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        tau=args.tau,
        utd=args.utd,
        device=device,
    )


def train_sac(
    env_id, seed, run_dir, protocol, config: SACConfig, train_env, periodic_eval_env,
    hidden_sizes=(256, 256), checkpoint_steps=(),
):
    from torchrl.objectives import SACLoss, SoftUpdate

    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    obs_dim = observation_spec.shape[-1]
    action_dim = action_spec.shape[-1]
    actor, qvalue = build_sac_modules(obs_dim, action_dim, action_spec, hidden_sizes)
    actor.to(config.device)
    qvalue.to(config.device)
    model_spec = {
        "observation_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": action_spec.space.low.tolist(),
        "action_high": action_spec.space.high.tolist(),
        "hidden_sizes": list(hidden_sizes),
    }
    loss_module = SACLoss(
        actor, qvalue, num_qvalue_nets=2, action_spec=action_spec,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=config.gamma)
    loss_module.to(config.device)
    target_updater = SoftUpdate(loss_module, tau=config.tau)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.learning_rate)
    replay = make_replay_buffer(config.buffer_size, config.batch_size)
    collector = make_collector(
        train_env, actor, config.n_envs, config.total_env_steps, config.device
    )
    logger = ProgressLogger(run_dir / "progress.csv")
    env_steps = 0
    next_eval = protocol.eval_freq_env_steps
    checkpoint_index = 0
    checkpoint_steps = sorted(checkpoint_steps)
    try:
        for batch in collector:
            collected = batch_env_steps(batch)
            env_steps += collected
            replay.extend(batch.reshape(-1).cpu())
            latest = {
                "loss_actor": float("nan"),
                "loss_qvalue": float("nan"),
                "loss_alpha": float("nan"),
                "alpha": float("nan"),
                "entropy": float("nan"),
            }
            if env_steps >= config.learning_starts and len(replay) >= config.batch_size:
                for _ in range(collected * config.utd):
                    losses = loss_module(replay.sample().to(config.device))
                    total_loss = sum(
                        value for key, value in losses.items()
                        if str(key).startswith("loss_")
                    )
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    target_updater.step()
                    latest = {
                        key: float(value.detach().mean()) for key, value in losses.items()
                    }
                collector.update_policy_weights_()
            logger.write({"env_steps": env_steps, "replay_size": len(replay), **latest})
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

    final_dir = save_checkpoint(
        run_dir / "final_model", actor=actor, model_spec=model_spec,
    )
    return TorchRLPolicyAdapter.load(final_dir, config.device), None
