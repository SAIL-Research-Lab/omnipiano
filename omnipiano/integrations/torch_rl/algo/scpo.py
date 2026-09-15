"""State-Conservative Policy Optimization (online SAC)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.replay import make_replay_buffer
from ..env.collectors import batch_env_steps, make_collector
from ..env.observation import observation_indices
from ..log.checkpoint import save_checkpoint
from omnipiano.integrations.eval_callback import EvalCallback
from ..log.logging import ProgressLogger, TrainingEpisodeTracker
from ..model.networks import build_sac_modules
from ..model.policy_adapter import TorchRLPolicyAdapter


@dataclass
class SCPOConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 24
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    learning_rate: float = 3e-4
    tau: float = 0.005
    utd: int = 1
    state_noise: float = 0.005
    gradient_observation: str = "dynamic"
    device: str = "cpu"


def make_config(args, device: str) -> SCPOConfig:
    return SCPOConfig(
        total_env_steps=args.total_steps, gamma=args.gamma, batch_size=args.batch_size,
        buffer_size=args.buffer_size, learning_starts=args.learning_starts,
        learning_rate=args.learning_rate, tau=args.tau, utd=args.utd,
        state_noise=args.scpo_state_noise,
        gradient_observation=args.scpo_gradient_observation, device=device,
    )


class SCPOLoss(torch.nn.Module):
    def __init__(self, actor, qvalue, action_spec, gamma, state_noise, state_indices):
        super().__init__()
        from torchrl.objectives import SACLoss

        self.loss = SACLoss(actor, qvalue, num_qvalue_nets=2, action_spec=action_spec,
                            delay_qvalue=True)
        self.loss.make_value_estimator(gamma=gamma)
        self.gamma = float(gamma)
        self.state_noise = float(state_noise)
        self.register_buffer("state_indices", state_indices)
        self.last_gradient_l1 = torch.tensor(0.0)
        self.last_penalty = torch.tensor(0.0)

    def _q(self, td, params):
        out = self.loss._vmap_qnetworkN0(
            td.select(*self.loss.qvalue_network.in_keys, strict=False), params
        )
        return out[self.loss.tensor_keys.state_action_value].min(0)[0].squeeze(-1)

    def _penalty(self, q, observation, create_graph):
        if self.state_noise == 0:
            return torch.zeros_like(q)
        gradient = torch.autograd.grad(q.sum(), observation, create_graph=create_graph)[0]
        selected = observation.index_select(-1, self.state_indices)
        scale = selected.detach().std(dim=0, unbiased=False).clamp_min(1e-6)
        # Convert dQ/d(raw observation) to dQ/d(standardized observation).
        norm = (gradient.index_select(-1, self.state_indices).abs() * scale).sum(-1)
        self.last_gradient_l1 = norm.detach().mean()
        penalty = self.state_noise * norm
        self.last_penalty = penalty.detach().mean()
        return penalty

    def actor_loss(self, td):
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from torchrl.objectives.sac import compute_log_prob

        observation = td["observation"].detach().requires_grad_(self.state_noise != 0)
        work = td.clone(False)
        work["observation"] = observation
        with set_exploration_type(ExplorationType.RANDOM), self.loss.actor_network_params.to_module(
            self.loss.actor_network, preserve_module_state=False
        ):
            dist = self.loss.actor_network.get_dist(work)
            action = dist.rsample()
        log_prob = compute_log_prob(dist, action, self.loss.tensor_keys.log_prob)
        work[self.loss.tensor_keys.action] = action
        q = self._q(work, self.loss._cached_detached_qvalue_params)
        conservative_q = q - self._penalty(q, observation, create_graph=True)
        return (self.loss._alpha * log_prob - conservative_q).mean(), {"log_prob": log_prob.detach()}

    def qvalue_loss(self, td):
        next_td = td["next"].copy()
        next_observation = next_td["observation"].detach().requires_grad_(self.state_noise != 0)
        next_td["observation"] = next_observation
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        from torchrl.objectives.sac import compute_log_prob

        actor_parameters = list(self.loss.actor_network.parameters())
        actor_requires_grad = [parameter.requires_grad for parameter in actor_parameters]
        for parameter in actor_parameters:
            parameter.requires_grad_(False)
        with set_exploration_type(ExplorationType.RANDOM), self.loss.actor_network_params.to_module(
            self.loss.actor_network, preserve_module_state=False
        ):
            dist = self.loss.actor_network.get_dist(next_td)
            next_action = dist.rsample()
            log_prob = compute_log_prob(dist, next_action, self.loss.tensor_keys.log_prob)
        for parameter, requires_grad in zip(actor_parameters, actor_requires_grad):
            parameter.requires_grad_(requires_grad)
        # The entire bootstrap target is constant for this optimization step.
        next_action = next_action.detach()
        log_prob = log_prob.detach()
        next_td[self.loss.tensor_keys.action] = next_action
        with torch.enable_grad():
            next_q = self._q(next_td, self.loss.target_qvalue_network_params)
            next_q = next_q - self._penalty(next_q, next_observation, create_graph=False)
            # The target branch must not backpropagate into the online update.
            # _penalty(..., create_graph=False) already consumed next_q's graph.
            next_q = next_q.detach()
        reward = td["next", "reward"].squeeze(-1)
        terminated = td["next", "terminated"].squeeze(-1).to(reward.dtype)
        target = reward + self.gamma * (1.0 - terminated) * (next_q - self.loss._alpha.detach() * log_prob)
        prediction = self.loss._vmap_qnetworkN0(
            td.select(*self.loss.qvalue_network.in_keys, strict=False), self.loss.qvalue_network_params
        )[self.loss.tensor_keys.state_action_value].squeeze(-1)
        return torch.nn.functional.smooth_l1_loss(prediction, target.expand_as(prediction), reduction="none").sum(0).mean()

    def forward(self, td):
        actor, meta = self.actor_loss(td)
        qvalue = self.qvalue_loss(td)
        alpha = self.loss.alpha_loss(meta["log_prob"]).mean()
        return {"loss_actor": actor, "loss_qvalue": qvalue, "loss_alpha": alpha,
                "alpha": self.loss._alpha, "entropy": -meta["log_prob"].mean()}


def train_scpo(env_id, seed, run_dir, protocol, config: SCPOConfig,
               train_env, periodic_eval_env, hidden_sizes=(256, 256), checkpoint_steps=()):
    from torchrl.objectives import SoftUpdate

    torch.manual_seed(seed)
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    observation_spec = train_env.observation_spec_unbatched["observation"]
    action_spec = train_env.action_spec_unbatched
    obs_dim, action_dim = observation_spec.shape[-1], action_spec.shape[-1]
    actor, qvalue = build_sac_modules(obs_dim, action_dim, action_spec, hidden_sizes)
    actor.to(config.device); qvalue.to(config.device)
    indices = observation_indices(periodic_eval_env, config.gradient_observation).to(config.device)
    loss_module = SCPOLoss(actor, qvalue, action_spec, config.gamma, config.state_noise, indices).to(config.device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.learning_rate)
    target_updater = SoftUpdate(loss_module.loss, tau=config.tau)
    replay = make_replay_buffer(config.buffer_size, config.batch_size)
    collector = make_collector(train_env, actor, config.n_envs, config.total_env_steps, config.device, config.learning_starts)
    model_spec = {"observation_dim": int(obs_dim), "action_dim": int(action_dim), "action_low": action_spec.space.low.tolist(), "action_high": action_spec.space.high.tolist(), "hidden_sizes": list(hidden_sizes)}
    logger = ProgressLogger(run_dir / "progress.csv", sb3_family="sac")
    episode_tracker = TrainingEpisodeTracker()
    update_steps = 0
    env_steps = 0; checkpoint_index = 0
    eval_callback = EvalCallback(
        env_id, periodic_eval_env, run_dir, seed + 10_000, protocol,
        policy_factory=lambda: TorchRLPolicyAdapter(actor, model_spec, config.device),
        save_best_fn=lambda path: save_checkpoint(path, actor=actor, model_spec=model_spec),
    )
    checkpoint_steps = sorted(checkpoint_steps)
    for batch in collector:
        collected = batch_env_steps(batch); env_steps += collected
        rollout_stats = episode_tracker.update(batch)
        replay.extend(batch.reshape(-1).cpu())
        latest = {"loss_actor": float("nan"), "loss_qvalue": float("nan"), "loss_alpha": float("nan"), "alpha": float("nan"), "entropy": float("nan")}
        if env_steps >= config.learning_starts and len(replay) >= config.batch_size:
            for _ in range(collected * config.utd):
                losses = loss_module(replay.sample().to(config.device))
                total = losses["loss_actor"] + losses["loss_qvalue"] + losses["loss_alpha"]
                optimizer.zero_grad(); total.backward(); optimizer.step(); target_updater.step()
                update_steps += 1
                latest = {key: float(value.detach().mean()) for key, value in losses.items()}
            collector.update_policy_weights_()
        eval_stats, did_eval = eval_callback.on_step(env_steps)
        logger.write({"env_steps": env_steps, **rollout_stats,
                      "replay_size": len(replay),
                      "learning_rate": config.learning_rate,
                      "n_updates": update_steps, **latest,
                      "state_gradient_l1": float(loss_module.last_gradient_l1),
                      "state_gradient_penalty": float(loss_module.last_penalty),
                      **eval_stats},
                     force=did_eval)
        while checkpoint_index < len(checkpoint_steps) and env_steps >= checkpoint_steps[checkpoint_index]:
            target = checkpoint_steps[checkpoint_index]; save_checkpoint(run_dir / "checkpoints" / f"checkpoint_{target}", actor=actor, model_spec=model_spec); checkpoint_index += 1
        if env_steps >= config.total_env_steps: break
    collector.shutdown()
    logger.close()
    final_dir = save_checkpoint(run_dir / "final_model", actor=actor, model_spec=model_spec)
    return TorchRLPolicyAdapter.load(final_dir, config.device), None
