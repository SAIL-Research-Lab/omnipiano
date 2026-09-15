"""Adaptive Adversarial Perturbation Soft Actor-Critic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..log.checkpoint import save_checkpoint
from omnipiano.integrations.eval_callback import EvalCallback
from ..log.logging import ProgressLogger, TrainingEpisodeTracker
from ..model.a2p_sac.adversarial import (
    A2PCollectionPolicy,
    AdaptiveCoefficient,
    mix_actions,
)
from ..model.networks import build_sac_modules, build_actor
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class A2PSACConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 24
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    adversary_learning_rate: float = 3e-4
    tau: float = 0.005
    utd: int = 1
    epsilon: float = 0.1
    adaptive_epsilon: bool = True
    epsilon_momentum: float = 0.5
    epsilon_learning_rate: float = 0.01
    epsilon_min: float = 0.03
    epsilon_max: float = 0.2
    epsilon_warmup_steps: int = 5_000
    adversary_update_frequency: int = 10
    device: str = "cpu"


def make_config(args, device: str) -> A2PSACConfig:
    adaptive = args.a2p_epsilon_mode == "adaptive"
    epsilon = 0.0 if args.a2p_epsilon_mode == "none" else min(max(args.a2p_epsilon, 0.0), 1.0)
    return A2PSACConfig(
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        adversary_learning_rate=args.adversary_learning_rate,
        tau=args.tau,
        utd=args.utd,
        epsilon=epsilon,
        adaptive_epsilon=adaptive,
        epsilon_warmup_steps=(
            args.learning_starts
            if args.a2p_warmup_steps is None
            else args.a2p_warmup_steps
        ),
        adversary_update_frequency=args.adversary_update_freq,
        device=device,
    )


class A2PSACLoss(torch.nn.Module):
    def __init__(self, actor, adversary, qvalue, action_spec, coefficient, gamma):
        super().__init__()
        from torchrl.objectives import SACLoss

        self.loss = SACLoss(
            actor, qvalue, num_qvalue_nets=2, action_spec=action_spec, delay_qvalue=True
        )
        self.loss.make_value_estimator(gamma=gamma)
        self.adversary = adversary
        self.coefficient = coefficient

    def _agent_action(self, tensordict):
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from torchrl.objectives.sac import compute_log_prob

        with set_exploration_type(ExplorationType.RANDOM), self.loss.actor_network_params.to_module(
            self.loss.actor_network, preserve_module_state=False
        ):
            distribution = self.loss.actor_network.get_dist(tensordict)
            action = distribution.rsample()
        log_prob = compute_log_prob(
            distribution, action, self.loss.tensor_keys.log_prob
        )
        return action, log_prob

    def _adversary_action(self, tensordict):
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        with set_exploration_type(ExplorationType.RANDOM):
            return self.adversary.get_dist(tensordict).rsample()

    def actor_loss(self, tensordict):
        weights = self.loss._maybe_get_priority_weight(tensordict)
        agent_action, log_prob = self._agent_action(tensordict)
        with torch.no_grad():
            adversary_action = self._adversary_action(tensordict)
        action = mix_actions(agent_action, adversary_action, self.coefficient.value)
        td_q = tensordict.select(*self.loss.qvalue_network.in_keys, strict=False)
        td_q.set(self.loss.tensor_keys.action, action)
        td_q = self.loss._vmap_qnetworkN0(td_q, self.loss._cached_detached_qvalue_params)
        qvalue = td_q[self.loss.tensor_keys.state_action_value].min(0)[0].squeeze(-1)
        actor_loss = (1.0 - self.coefficient.value) * self.loss._alpha * log_prob - qvalue
        return self.loss._reduce_loss(actor_loss, tensordict=tensordict, weights=weights), {
            "log_prob": log_prob.detach()
        }

    def _target(self, tensordict):
        next_tensordict = tensordict["next"].copy()
        with torch.no_grad():
            agent_action, log_prob = self._agent_action(next_tensordict)
            adversary_action = self._adversary_action(next_tensordict)
            action = mix_actions(agent_action, adversary_action, self.coefficient.value)
            next_tensordict[self.loss.tensor_keys.action] = action
            next_tensordict = self.loss._vmap_qnetworkN0(
                next_tensordict, self.loss.target_qvalue_network_params
            )
            next_value = next_tensordict[self.loss.tensor_keys.state_action_value]
            if next_value.shape[-len(log_prob.shape):] != log_prob.shape:
                log_prob = log_prob.unsqueeze(-1)
            next_value = (
                next_value
                - (1.0 - self.coefficient.value) * self.loss._alpha * log_prob
            ).min(0)[0]
            target_td = tensordict.clone(False)
            target_td[("next", self.loss.value_estimator.tensor_keys.value)] = next_value
            return self.loss.value_estimator.value_estimate(target_td).squeeze(-1)

    def qvalue_loss(self, tensordict):
        weights = self.loss._maybe_get_priority_weight(tensordict)
        target = self._target(tensordict)
        expanded = self.loss._vmap_qnetworkN0(
            tensordict.select(*self.loss.qvalue_network.in_keys, strict=False),
            self.loss.qvalue_network_params,
        )
        prediction = expanded[self.loss.tensor_keys.state_action_value].squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(
            prediction, target.expand_as(prediction), reduction="none"
        ).sum(0)
        return self.loss._reduce_loss(loss, tensordict=tensordict, weights=weights)

    def forward(self, tensordict):
        actor_loss, metadata = self.actor_loss(tensordict)
        qvalue_loss = self.qvalue_loss(tensordict)
        alpha_loss = self.loss.alpha_loss(metadata["log_prob"]).mean()
        return {
            "loss_actor": actor_loss,
            "loss_qvalue": qvalue_loss,
            "loss_alpha": alpha_loss,
            "alpha": self.loss._alpha,
            "entropy": -metadata["log_prob"].mean(),
        }

    def adversary_loss(self, tensordict):
        with torch.no_grad():
            agent_action, _ = self._agent_action(tensordict)
        adversary_action = self._adversary_action(tensordict)
        action = mix_actions(agent_action, adversary_action, self.coefficient.value)
        td_q = tensordict.select(*self.loss.qvalue_network.in_keys, strict=False)
        td_q[self.loss.tensor_keys.action] = action
        td_q = self.loss._vmap_qnetworkN0(td_q, self.loss._cached_detached_qvalue_params)
        return td_q[self.loss.tensor_keys.state_action_value].min(0)[0].mean()


def train_a2p_sac(
    env_id, seed, run_dir, protocol, config: A2PSACConfig,
    train_env, periodic_eval_env, hidden_sizes=(256, 256), checkpoint_steps=(),
):
    from torchrl.objectives import SoftUpdate

    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    obs_dim = observation_spec.shape[-1]
    action_dim = action_spec.shape[-1]
    actor, qvalue = build_sac_modules(obs_dim, action_dim, action_spec, hidden_sizes)
    adversary = build_actor(obs_dim, action_dim, action_spec, hidden_sizes)
    actor.to(config.device)
    qvalue.to(config.device)
    adversary.to(config.device)
    coefficient = AdaptiveCoefficient(
        config.epsilon, config.adaptive_epsilon, config.epsilon_momentum,
        config.epsilon_learning_rate, config.epsilon_min, config.epsilon_max,
        config.epsilon_warmup_steps,
    )
    collection_policy = A2PCollectionPolicy(actor, adversary, coefficient)
    loss_module = A2PSACLoss(
        actor, adversary, qvalue, action_spec, coefficient, config.gamma
    ).to(config.device)
    target_updater = SoftUpdate(loss_module.loss, tau=config.tau)
    adversary_ids = {id(parameter) for parameter in adversary.parameters()}
    sac_optimizer = torch.optim.Adam(
        [parameter for parameter in loss_module.parameters() if id(parameter) not in adversary_ids],
        lr=config.learning_rate,
    )
    adversary_optimizer = torch.optim.Adam(
        adversary.parameters(), lr=config.adversary_learning_rate
    )
    replay = make_replay_buffer(config.buffer_size, config.batch_size)
    collector = make_collector(
        train_env, collection_policy, config.n_envs, config.total_env_steps,
        config.device, config.learning_starts,
    )
    logger = ProgressLogger(run_dir / "progress.csv", sb3_family="sac")
    episode_tracker = TrainingEpisodeTracker()
    model_spec = {
        "observation_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": action_spec.space.low.tolist(),
        "action_high": action_spec.space.high.tolist(),
        "hidden_sizes": list(hidden_sizes),
    }
    env_steps = 0
    update_steps = 0
    eval_callback = EvalCallback(
        env_id, periodic_eval_env, run_dir, seed + 10_000, protocol,
        policy_factory=lambda: TorchRLPolicyAdapter(actor, model_spec, config.device),
        save_best_fn=lambda path: save_checkpoint(path, actor=actor, model_spec=model_spec),
    )
    checkpoint_index = 0
    checkpoint_steps = sorted(checkpoint_steps)
    try:
        for batch in collector:
            collected = batch_env_steps(batch)
            env_steps += collected
            rollout_stats = episode_tracker.update(batch)
            replay.extend(batch.reshape(-1).cpu())
            latest = {
                "loss_actor": float("nan"), "loss_qvalue": float("nan"),
                "loss_alpha": float("nan"), "loss_adversary": float("nan"),
                "alpha": float("nan"), "entropy": float("nan"),
            }
            if env_steps >= config.learning_starts and len(replay) >= config.batch_size:
                for _ in range(collected * config.utd):
                    sample = replay.sample().to(config.device)
                    adversary_step = (
                        (update_steps + 1) % config.adversary_update_frequency == 0
                    )
                    if adversary_step:
                        adversary_loss = loss_module.adversary_loss(sample)
                        adversary_optimizer.zero_grad()
                        adversary_loss.backward()
                        adversary_optimizer.step()
                        latest["loss_adversary"] = float(adversary_loss.detach())
                        losses = {}
                    else:
                        losses = loss_module(sample)
                        sac_loss = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_alpha"]
                        sac_optimizer.zero_grad()
                        sac_loss.backward()
                        sac_optimizer.step()
                        target_updater.step()
                    update_steps += 1
                    latest.update({key: float(value.detach().mean()) for key, value in losses.items()})
                collector.update_policy_weights_()
            diagnostics = {
                "epsilon": coefficient.value,
                "action_distance": coefficient.moving_distance,
                "agent_action_l2": float(batch.get("agent_action", batch["action"]).norm(dim=-1).mean()),
                "adversary_action_l2": float(batch.get("adversary_action", batch["action"]).norm(dim=-1).mean()),
                "executed_delta_l2": float(batch.get("executed_delta", torch.zeros_like(batch["action"])).norm(dim=-1).mean()),
                "sustain_adversarial_delta": float(batch.get("executed_delta", torch.zeros_like(batch["action"]))[..., -1].abs().mean()),
            }
            eval_stats, did_eval = eval_callback.on_step(env_steps)
            logger.write({
                "env_steps": env_steps,
                **rollout_stats,
                "replay_size": len(replay),
                "learning_rate": config.learning_rate,
                "n_updates": update_steps,
                **latest,
                **diagnostics,
                **eval_stats,
            }, force=did_eval)
            while checkpoint_index < len(checkpoint_steps) and env_steps >= checkpoint_steps[checkpoint_index]:
                target = checkpoint_steps[checkpoint_index]
                save_checkpoint(
                    run_dir / "checkpoints" / f"checkpoint_{target}", actor=actor,
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
