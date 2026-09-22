"""Native, feed-forward MARL learners.

No Ray dependency. All batches contain aligned JOINT environment transitions.
HAPPO/A2PO/central PPO use raw Gaussian actions with execution-time clipping.
MAT masks heterogeneous action padding before conditioning and loss reduction.
"""
from __future__ import annotations

import copy
import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from omnipiano.multiagent.algos._a2po_math import (
    adaptive_clip,
    a2po_surrogate,
    clipped_preceding_ratio,
    official_order_scores,
    preopc_advantages,
    semi_greedy_order,
)
from omnipiano.multiagent.algos._facmac_mixer import FactoredMixer
from omnipiano.multiagent.algos._happo_math import (
    compound_log_factor_update,
    happo_surrogate,
)


def ranges(widths):
    ends = np.cumsum([0, *widths]).tolist()
    return [slice(ends[i], ends[i + 1]) for i in range(len(widths))]


def mlp(inp, out, args, output_gain=None):
    layers = []
    if args.input_layer_norm:
        layers.append(nn.LayerNorm(inp))
    activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[args.activation]
    previous = inp
    for width in args.hidden_sizes_parsed:
        layer = nn.Linear(previous, width)
        nn.init.orthogonal_(layer.weight, args.hidden_orthogonal_gain)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, activation()))
        previous = width
    layer = nn.Linear(previous, out)
    nn.init.orthogonal_(
        layer.weight,
        args.value_output_gain if output_gain is None else output_gain,
    )
    nn.init.zeros_(layer.bias)
    layers.append(layer)
    return nn.Sequential(*layers)


def facmac_mlp(inp, out, hidden_sizes):
    """Official FACMAC MAMuJoCo MLP shape with PyTorch default init.

    The reference uses two 400-wide ReLU layers for both actor and per-agent
    utility.  It does not apply LayerNorm or orthogonal initialisation.
    ``hidden_sizes`` remains explicit so every realized architecture is stored
    in the experiment JSON/checkpoint identity.
    """
    layers = []
    previous = inp
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, out))
    return nn.Sequential(*layers)


class ValueNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enabled = args.value_norm
        self.beta = args.value_norm_beta
        self.epsilon = args.value_norm_epsilon
        self.floor = args.value_norm_variance_floor
        for key in ("mean_acc", "square_acc", "weight"):
            self.register_buffer(key, torch.zeros(()))

    def statistics(self):
        weight = self.weight.clamp_min(self.epsilon)
        mean = self.mean_acc / weight
        variance = (self.square_acc / weight - mean.square()).clamp_min(self.floor)
        return mean, variance.sqrt()

    @torch.no_grad()
    def update(self, values):
        if self.enabled:
            values = values.detach()
            self.mean_acc.mul_(self.beta).add_(values.mean() * (1 - self.beta))
            self.square_acc.mul_(self.beta).add_(
                values.square().mean() * (1 - self.beta))
            self.weight.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, values):
        if not self.enabled:
            return values
        mean, std = self.statistics()
        return (values - mean) / std

    def denormalize(self, values):
        if not self.enabled:
            return values
        mean, std = self.statistics()
        return values * std + mean


class Gaussian(nn.Module):
    def __init__(self, inp, dim, args, squashed=False):
        super().__init__()
        self.dim, self.squashed = dim, squashed
        self.minimum, self.maximum = args.log_std_min, args.log_std_max
        self.net = mlp(
            inp, dim * (2 if squashed else 1), args, args.policy_output_gain)
        if squashed:
            self.log_std = None
            with torch.no_grad():
                self.net[-1].bias[dim:].fill_(args.initial_log_std)
        else:
            self.log_std = nn.Parameter(torch.full((dim,), args.initial_log_std))

    def distribution(self, x):
        prediction = self.net(x)
        if self.squashed:
            mean, log_std = prediction.chunk(2, -1)
        else:
            mean, log_std = prediction, self.log_std.expand_as(prediction)
        return Normal(mean, log_std.clamp(self.minimum, self.maximum).exp())

    def sample(self, x, deterministic=False):
        distribution = self.distribution(x)
        raw = distribution.mean if deterministic else distribution.rsample()
        logp = distribution.log_prob(raw)
        if self.squashed:
            # Stable log(1 - tanh(raw)^2), before summing valid dimensions.
            logp = logp - 2 * (math.log(2) - raw - F.softplus(-2 * raw))
            return raw.tanh(), logp.sum(-1)
        return raw, logp.sum(-1)


def _assert_finite_async(value, message):
    """Queue a CUDA-side finite check without synchronizing the host."""
    check = torch.isfinite(value).all()
    if value.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(check, message)
    elif not bool(check):
        raise FloatingPointError(message)


def optimize(optimizer, loss, parameters, maximum_norm, *, sync_stats=True):
    parameters = list(parameters)
    if sync_stats:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite native learner loss")
    else:
        _assert_finite_async(loss, "Non-finite native learner loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(
        parameters, maximum_norm, error_if_nonfinite=sync_stats)
    if not sync_stats:
        _assert_finite_async(norm, "Non-finite native learner gradient norm")
    optimizer.step()
    if sync_stats:
        return float(loss.detach()), float(norm.detach())
    return loss.detach(), norm.detach()


def value_loss(prediction, old_prediction, target, normalizer, clip):
    normalizer.update(target)
    normalized_target = normalizer.normalize(target)
    clipped = old_prediction + (prediction - old_prediction).clamp(-clip, clip)
    return 0.5 * torch.maximum(
        (prediction - normalized_target).square(),
        (clipped - normalized_target).square(),
    ).mean()


@torch.no_grad()
def soft_update(target, source, tau):
    destination_parameters = list(target.parameters())
    source_parameters = list(source.parameters())
    if len(destination_parameters) != len(source_parameters):
        raise ValueError("Target/source parameter layouts differ")
    if destination_parameters:
        torch._foreach_lerp_(destination_parameters, source_parameters, tau)
    destination_buffers = list(target.buffers())
    source_buffers = list(source.buffers())
    if len(destination_buffers) != len(source_buffers):
        raise ValueError("Target/source buffer layouts differ")
    if destination_buffers:
        torch._foreach_copy_(destination_buffers, source_buffers)


class NativeModel(nn.Module):
    on_policy = False

    def __init__(self, args, meta, options):
        super().__init__()
        self.args, self.meta, self.options = args, meta, options
        self.n = len(meta["agents"])
        self.od = [hi - lo for lo, hi in meta["own_slices"]]
        self.ad = list(meta["action_dims"])
        self.os, self.ac = ranges(self.od), ranges(self.ad)
        self.sd = int(meta["state_dim"])
        self.optimizers = {}

    def local(self, batch, index):
        return batch["o"][:, self.os[index]]

    def minibatches(self, size):
        for _ in range(self.args.num_epochs):
            yield from self.epoch_minibatches(size)

    def epoch_minibatches(self, size):
        """Yield one independently shuffled epoch of minibatch indices."""
        order = torch.randperm(size, device=next(self.parameters()).device)
        for start in range(0, size, self.args.minibatch_size):
            yield order[start:start + self.args.minibatch_size]

    def optimizer_state(self):
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def restore_optimizers(self, state):
        if set(state) != set(self.optimizers):
            raise ValueError("Checkpoint optimizer set mismatch")
        for name, value in state.items():
            self.optimizers[name].load_state_dict(value)


class PPO(NativeModel):
    """Strict HAPPO, or the one-policy centralized PPO control.

    HAPPO uses one independent actor per agent and one shared ``V(s)`` critic.
    The shared critic is HARL's environment-provided-state (EP) formulation:
    OmniPiano exposes one identical physical global state to every agent and a
    shared team reward, so neither agent identity nor duplicated critics belong
    in the value input.  The critic is deliberately updated only after every
    actor has completed its sequential update.
    """
    on_policy = True

    def __init__(self, args, meta, options):
        super().__init__(args, meta, options)
        self.central = args.algo == "ppo-monolithic"
        self.nactors = 1 if self.central else self.n
        actor_inputs = [self.sd] if self.central else self.od
        action_dims = [sum(self.ad)] if self.central else self.ad
        self.actors = nn.ModuleList(
            Gaussian(inp, dim, args) for inp, dim in zip(actor_inputs, action_dims))
        self.critic = mlp(self.sd, 1, args)
        self.normalizer = ValueNorm(args)

    def build_optimizers(self):
        for i in range(self.nactors):
            self.optimizers[f"actor_{i}"] = torch.optim.Adam(
                self.actors[i].parameters(), lr=self.args.lr,
                eps=self.args.adam_epsilon)
        self.optimizers["critic"] = torch.optim.Adam(
            self.critic.parameters(), lr=self.args.critic_lr,
            eps=self.args.adam_epsilon)

    def actor_input(self, batch, i):
        return batch["s"] if self.central else self.local(batch, i)

    def values(self, batch):
        normalized = self.critic(batch["s"]).squeeze(-1)
        values = self.normalizer.denormalize(normalized)
        # The rollout transport stores one value column per environment agent.
        # EP HAPPO has one shared value, so columns are identical by design.
        columns = 1 if self.central else self.n
        return (
            normalized[:, None].expand(-1, columns),
            values[:, None].expand(-1, columns),
        )

    def act(self, batch, deterministic=False):
        pairs = [
            actor.sample(self.actor_input(batch, i), deterministic)
            for i, actor in enumerate(self.actors)
        ]
        vn, values = self.values(batch)
        return {
            "a": torch.cat([pair[0] for pair in pairs], -1),
            "lp": torch.stack([pair[1] for pair in pairs], -1),
            "vn": vn, "v": values,
        }

    def evaluate_actor(self, batch, i):
        actions = batch["a"] if self.central else batch["a"][:, self.ac[i]]
        distribution = self.actors[i].distribution(self.actor_input(batch, i))
        return (
            distribution.log_prob(actions).sum(-1),
            distribution.entropy().sum(-1),
        )

    def agent_update_order(self):
        """Sample one fresh HAPPO permutation (split out for audit tests)."""
        return torch.randperm(self.nactors).tolist()

    def learn(self, batch):
        args = self.args
        size = len(batch["a"])
        if batch["lp"].shape != (size, self.nactors):
            raise ValueError("HAPPO rollout log-probability layout changed")
        expected_value_shape = (size, 1 if self.central else self.n)
        for name in ("vn", "v", "nv", "adv", "ret"):
            if batch[name].shape != expected_value_shape:
                raise ValueError(
                    f"HAPPO rollout {name!r} has shape {tuple(batch[name].shape)}, "
                    f"expected {expected_value_shape}"
                )

        # Environment-provided global state + shared reward => one advantage.
        # Repeated rollout columns must remain bit-identical; accepting drift
        # would silently turn EP HAPPO into an undocumented FP variant.
        shared_advantage = batch["adv"][:, 0]
        shared_return = batch["ret"][:, 0]
        if not self.central:
            for name, value in (("adv", batch["adv"]), ("ret", batch["ret"])):
                reference = value[:, :1].expand_as(value)
                if not torch.equal(value, reference):
                    raise RuntimeError(
                        f"EP HAPPO requires shared {name} columns across agents"
                    )
        advantages = (
            shared_advantage - shared_advantage.mean()
        ) / (shared_advantage.std(unbiased=False) + 1e-5)

        order = self.agent_update_order()
        if sorted(order) != list(range(self.nactors)):
            raise RuntimeError("HAPPO agent_update_order is not a permutation")
        log_factor = torch.zeros(size, device=batch["a"].device)
        stats = {}
        all_actor_losses, all_actor_norms, all_entropies = [], [], []

        for position, i in enumerate(order):
            with torch.no_grad():
                before, _ = self.evaluate_actor(batch, i)
                before = before.clone()
                if not torch.allclose(
                    before, batch["lp"][:, i], atol=1e-4, rtol=1e-4
                ):
                    raise RuntimeError("Stale rollout or inconsistent stored-action logp")
            factor = log_factor.exp()
            stats[f"actor_{i}_update_position"] = float(position)
            stats[f"actor_{i}_factor_mean"] = float(factor.mean())
            stats[f"actor_{i}_factor_min"] = float(factor.min())
            stats[f"actor_{i}_factor_max"] = float(factor.max())

            actor_losses, actor_norms, entropies, ratios = [], [], [], []

            for indices in self.minibatches(size):
                minibatch = {k: v[indices] for k, v in batch.items()}
                logp, entropy = self.evaluate_actor(minibatch, i)
                surrogate = happo_surrogate(
                    logp,
                    minibatch["lp"][:, i],
                    advantages[indices],
                    log_factor[indices].detach(),
                    args.clip_param,
                )
                loss = -surrogate.mean() - args.entropy_coeff * entropy.mean()
                loss_value, norm = optimize(
                    self.optimizers[f"actor_{i}"], loss,
                    self.actors[i].parameters(), args.grad_clip)
                actor_losses.append(loss_value)
                actor_norms.append(norm)
                entropies.append(float(entropy.detach().mean()))
                ratios.append(float(
                    (logp.detach() - minibatch["lp"][:, i]).exp().mean()
                ))

            updates = len(actor_losses)
            stats[f"actor_{i}_loss"] = sum(actor_losses) / updates
            stats[f"actor_{i}_grad_norm"] = sum(actor_norms) / updates
            stats[f"actor_{i}_entropy"] = sum(entropies) / updates
            stats[f"actor_{i}_ratio"] = sum(ratios) / updates
            all_actor_losses.append(stats[f"actor_{i}_loss"])
            all_actor_norms.append(stats[f"actor_{i}_grad_norm"])
            all_entropies.append(stats[f"actor_{i}_entropy"])

            with torch.no_grad():
                after, _ = self.evaluate_actor(batch, i)
                log_factor = compound_log_factor_update(
                    log_factor, after, before)

        critic_losses, critic_norms = [], []
        for indices in self.minibatches(size):
            prediction = self.critic(batch["s"][indices]).squeeze(-1)
            loss = args.vf_loss_coeff * value_loss(
                prediction,
                batch["vn"][indices, 0],
                shared_return[indices],
                self.normalizer,
                args.vf_clip_param,
            )
            value, norm = optimize(
                self.optimizers["critic"], loss,
                self.critic.parameters(), args.grad_clip)
            critic_losses.append(value)
            critic_norms.append(norm)
        updates = len(critic_losses)
        stats["critic_loss"] = sum(critic_losses) / updates
        stats["critic_grad_norm"] = sum(critic_norms) / updates
        stats["actor_loss"] = sum(all_actor_losses) / len(all_actor_losses)
        stats["actor_grad_norm"] = sum(all_actor_norms) / len(all_actor_norms)
        stats["entropy"] = sum(all_entropies) / len(all_entropies)
        stats["compound_factor_abs_log_mean"] = float(log_factor.abs().mean())
        stats["compound_factor_abs_log_max"] = float(log_factor.abs().max())
        return stats


class A2PO(NativeModel):
    """Canonical A2PO on aligned joint rollouts.

    The non-shared continuous-control path in the public A2PO repository owns
    one actor and one global-state critic per agent. That is also the only
    topology compatible with OmniPiano's heterogeneous action sizes without
    padding policy outputs. PreOPC targets are detached return estimators;
    gradients flow only through the current actor and its own critic.
    """

    on_policy = True

    def __init__(self, args, meta, options):
        super().__init__(args, meta, options)
        self.actors = nn.ModuleList(
            Gaussian(inp, dim, args) for inp, dim in zip(self.od, self.ad)
        )
        self.critics = nn.ModuleList(mlp(self.sd, 1, args) for _ in range(self.n))
        self.normalizers = nn.ModuleList(ValueNorm(args) for _ in range(self.n))

    def build_optimizers(self):
        for i in range(self.n):
            self.optimizers[f"actor_{i}"] = torch.optim.Adam(
                self.actors[i].parameters(), lr=self.args.lr,
                eps=self.args.adam_epsilon)
            self.optimizers[f"critic_{i}"] = torch.optim.Adam(
                self.critics[i].parameters(), lr=self.args.critic_lr,
                eps=self.args.adam_epsilon)

    def values(self, batch):
        normalized = torch.stack([
            critic(batch["s"]).squeeze(-1) for critic in self.critics
        ], dim=-1)
        values = torch.stack([
            normalizer.denormalize(normalized[:, i])
            for i, normalizer in enumerate(self.normalizers)
        ], dim=-1)
        return normalized, values

    def act(self, batch, deterministic=False):
        pairs = [
            actor.sample(self.local(batch, i), deterministic)
            for i, actor in enumerate(self.actors)
        ]
        normalized, values = self.values(batch)
        return {
            "a": torch.cat([pair[0] for pair in pairs], dim=-1),
            "lp": torch.stack([pair[1] for pair in pairs], dim=-1),
            "vn": normalized,
            "v": values,
        }

    def evaluate_actor(self, batch, index):
        distribution = self.actors[index].distribution(self.local(batch, index))
        actions = batch["a"][:, self.ac[index]]
        return (
            distribution.log_prob(actions).sum(-1),
            distribution.entropy().sum(-1),
        )

    def agent_update_order(self, advantages, values):
        scores = official_order_scores(
            advantages, values, self.options["order_score_epsilon"])
        return semi_greedy_order(scores.detach().cpu().numpy())

    def _critic_pass(self, batch, agent, targets):
        losses, norms = [], []
        for indices in self.epoch_minibatches(len(targets)):
            prediction = self.critics[agent](batch["s"][indices]).squeeze(-1)
            loss = self.args.vf_loss_coeff * value_loss(
                prediction,
                batch["vn"][indices, agent],
                targets[indices],
                self.normalizers[agent],
                self.args.vf_clip_param,
            )
            value, norm = optimize(
                self.optimizers[f"critic_{agent}"], loss,
                self.critics[agent].parameters(), self.args.grad_clip)
            losses.append(value)
            norms.append(norm)
        return losses, norms

    def learn(self, batch):
        args, options = self.args, self.options
        size = len(batch["a"])
        expected = (size, self.n)
        for name in ("lp", "vn", "v", "nv", "adv", "ret"):
            if batch[name].shape != expected:
                raise ValueError(
                    f"A2PO rollout {name!r} has shape {tuple(batch[name].shape)}, "
                    f"expected {expected}"
                )
        for name in ("r", "term", "trunc"):
            if batch[name].shape != (size,):
                raise ValueError(f"A2PO rollout {name!r} must have shape {(size,)}")
        if batch["keys"].shape != (size, 3):
            raise ValueError("A2PO requires [stream, episode, timestep] rollout keys")

        with torch.no_grad():
            for i in range(self.n):
                current, _ = self.evaluate_actor(batch, i)
                if not torch.allclose(
                    current, batch["lp"][:, i], atol=1e-4, rtol=1e-4
                ):
                    raise RuntimeError(
                        "Stale rollout or inconsistent stored-action logp"
                    )

        # The upstream semi-greedy rule scores globally standardized rollout
        # advantages against the ValueNorm-space predictions stored at sample
        # time. It deliberately does not use denormalized physical returns.
        order = self.agent_update_order(batch["adv"], batch["vn"])
        if sorted(order) != list(range(self.n)):
            raise RuntimeError("A2PO agent_update_order is not a permutation")

        preceding_log_ratio = torch.zeros(
            size, device=batch["a"].device, dtype=batch["a"].dtype)
        stats = {}
        all_actor_losses, all_actor_norms, all_entropies = [], [], []
        all_critic_losses, all_critic_norms = [], []
        all_pretrain_losses = []
        working_advantages = batch["adv"].clone()

        for position, agent in enumerate(order):
            with torch.no_grad():
                before, _ = self.evaluate_actor(batch, agent)
                if not torch.allclose(
                    before, batch["lp"][:, agent], atol=1e-4, rtol=1e-4
                ):
                    raise RuntimeError(
                        "A2PO current actor changed before its ordered update"
                    )
                corrected_advantage = preopc_advantages(
                    batch["r"], batch["v"][:, agent], batch["nv"][:, agent],
                    batch["term"], batch["trunc"], preceding_log_ratio,
                    batch["keys"], gamma=args.gamma,
                    trace_lambda=args.gae_lambda,
                    trace_clip_param=options["trace_clip_param"],
                )
                if position == 0 and not torch.allclose(
                    corrected_advantage, batch["adv"][:, agent],
                    atol=1e-5, rtol=1e-5,
                ):
                    raise RuntimeError(
                        "A2PO PreOPC with no preceding agents did not reduce to GAE"
                    )
                corrected_return = corrected_advantage + batch["v"][:, agent]
                working_advantages[:, agent] = corrected_advantage
                normalized_all = (
                    working_advantages - working_advantages.mean()
                ) / (working_advantages.std(unbiased=False) + 1e-5)
                normalized_advantage = normalized_all[:, agent]
                preceding = clipped_preceding_ratio(
                    preceding_log_ratio, options["preceding_ratio_clip"])

            adaptive = adaptive_clip(
                args.clip_param, position + 1, self.n,
                options["adaptive_clip_weight"])
            stats[f"actor_{agent}_update_position"] = float(position)
            stats[f"actor_{agent}_adaptive_clip"] = float(adaptive)
            stats[f"actor_{agent}_preceding_ratio_mean"] = float(preceding.mean())
            stats[f"actor_{agent}_preceding_ratio_min"] = float(preceding.min())
            stats[f"actor_{agent}_preceding_ratio_max"] = float(preceding.max())
            stats[f"actor_{agent}_preopc_adv_mean"] = float(
                corrected_advantage.mean())
            stats[f"actor_{agent}_preopc_adv_std"] = float(
                corrected_advantage.std(unbiased=False))

            actor_losses, actor_norms, entropies, ratios = [], [], [], []
            critic_losses, critic_norms = [], []
            pre_losses, pre_norms = [], []
            # The upstream agent-loop-first path executes both stages inside
            # every PPO epoch: one critic-only epoch, immediately followed by
            # one actor+critic epoch. Running every critic-only epoch up front
            # would produce a different optimizer trajectory.
            for _ in range(args.num_epochs):
                stage_losses, stage_norms = self._critic_pass(
                    batch, agent, corrected_return)
                pre_losses.extend(stage_losses)
                pre_norms.extend(stage_norms)

                for indices in self.epoch_minibatches(size):
                    minibatch = {
                        key: value[indices] for key, value in batch.items()
                    }
                    logp, entropy = self.evaluate_actor(minibatch, agent)
                    surrogate = a2po_surrogate(
                        logp,
                        minibatch["lp"][:, agent],
                        normalized_advantage[indices],
                        preceding_log_ratio[indices],
                        adaptive,
                        options["preceding_ratio_clip"],
                    )
                    actor_loss = (
                        -surrogate.mean()
                        - args.entropy_coeff * entropy.mean()
                    )
                    loss_value, norm = optimize(
                        self.optimizers[f"actor_{agent}"], actor_loss,
                        self.actors[agent].parameters(), args.grad_clip)
                    actor_losses.append(loss_value)
                    actor_norms.append(norm)
                    entropies.append(float(entropy.detach().mean()))
                    ratios.append(float(
                        (logp.detach() - minibatch["lp"][:, agent])
                        .exp().mean()
                    ))

                    prediction = self.critics[agent](
                        minibatch["s"]).squeeze(-1)
                    critic_loss = args.vf_loss_coeff * value_loss(
                        prediction,
                        minibatch["vn"][:, agent],
                        corrected_return[indices],
                        self.normalizers[agent],
                        args.vf_clip_param,
                    )
                    value, critic_norm = optimize(
                        self.optimizers[f"critic_{agent}"], critic_loss,
                        self.critics[agent].parameters(), args.grad_clip)
                    critic_losses.append(value)
                    critic_norms.append(critic_norm)

            stats[f"critic_{agent}_pretrain_loss"] = (
                sum(pre_losses) / len(pre_losses)
            )
            stats[f"critic_{agent}_pretrain_grad_norm"] = (
                sum(pre_norms) / len(pre_norms)
            )
            all_pretrain_losses.extend(pre_losses)

            updates = len(actor_losses)
            stats[f"actor_{agent}_loss"] = sum(actor_losses) / updates
            stats[f"actor_{agent}_grad_norm"] = sum(actor_norms) / updates
            stats[f"actor_{agent}_entropy"] = sum(entropies) / updates
            stats[f"actor_{agent}_ratio"] = sum(ratios) / updates
            stats[f"critic_{agent}_loss"] = sum(critic_losses) / updates
            stats[f"critic_{agent}_grad_norm"] = sum(critic_norms) / updates
            all_actor_losses.extend(actor_losses)
            all_actor_norms.extend(actor_norms)
            all_entropies.extend(entropies)
            all_critic_losses.extend(critic_losses)
            all_critic_norms.extend(critic_norms)

            with torch.no_grad():
                after, _ = self.evaluate_actor(batch, agent)
                preceding_log_ratio = (
                    preceding_log_ratio
                    + after.detach()
                    - batch["lp"][:, agent].detach()
                ).detach()
                if not bool(torch.isfinite(preceding_log_ratio).all()):
                    raise FloatingPointError(
                        "Non-finite A2PO preceding-agent log ratio")

        stats["actor_loss"] = sum(all_actor_losses) / len(all_actor_losses)
        stats["actor_grad_norm"] = sum(all_actor_norms) / len(all_actor_norms)
        stats["entropy"] = sum(all_entropies) / len(all_entropies)
        stats["critic_loss"] = sum(all_critic_losses) / len(all_critic_losses)
        stats["critic_grad_norm"] = sum(all_critic_norms) / len(all_critic_norms)
        stats["critic_pretrain_loss"] = (
            sum(all_pretrain_losses) / len(all_pretrain_losses))
        stats["preceding_log_ratio_abs_mean"] = float(
            preceding_log_ratio.abs().mean())
        stats["preceding_log_ratio_abs_max"] = float(
            preceding_log_ratio.abs().max())
        return stats


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        if decoder:
            self.cross_attention = nn.MultiheadAttention(
                width, heads, dropout=0.0, batch_first=True)
            self.norm3 = nn.LayerNorm(width)

    def forward(self, x, representation=None, mask=None):
        attended = self.self_attention(
            x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm1(x + attended)
        if self.decoder:
            attended = self.cross_attention(
                representation, x, x, attn_mask=mask, need_weights=False)[0]
            x = self.norm2(representation + attended)
            return self.norm3(x + self.ff(x))
        return self.norm2(x + self.ff(x))


class MAT(NativeModel):
    on_policy = True

    def __init__(self, args, meta, options):
        super().__init__(args, meta, options)
        width, heads, blocks = (
            options["embed_dim"], options["heads"], options["blocks"])
        self.nvalues = self.n
        self.maximum_action_dim = max(self.ad)
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU())
            for d in self.od)
        self.input_norm = nn.LayerNorm(width)
        self.encoder = nn.ModuleList(
            AttentionBlock(width, heads) for _ in range(blocks))
        self.action_embedding = nn.Sequential(
            nn.Linear(self.maximum_action_dim, width),
            nn.GELU(), nn.LayerNorm(width))
        self.decoder = nn.ModuleList(
            AttentionBlock(width, heads, decoder=True) for _ in range(blocks))

        def head(output):
            return nn.Sequential(
                nn.Linear(width, width), nn.GELU(),
                nn.LayerNorm(width), nn.Linear(width, output))

        self.value_head = head(1)
        self.mean_head = head(self.maximum_action_dim)
        self.std_parameter = nn.Parameter(torch.full(
            (self.maximum_action_dim,), float(options["std_parameter_init"])))
        self.normalizer = ValueNorm(args)
        self.register_buffer(
            "causal_mask", torch.ones(self.n, self.n, dtype=torch.bool).triu(1))
        self.register_buffer(
            "action_mask",
            torch.arange(self.maximum_action_dim)[None, :]
            < torch.tensor(self.ad)[:, None],
        )
        # An explicit initialization for this adaptation; not a bitwise oracle.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.MultiheadAttention):
                nn.init.orthogonal_(module.in_proj_weight, gain=0.01)
                nn.init.zeros_(module.in_proj_bias)

    def build_optimizers(self):
        self.optimizers["joint"] = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, eps=self.args.adam_epsilon)

    def encode(self, batch):
        tokens = torch.stack([
            adapter(self.local(batch, i))
            for i, adapter in enumerate(self.adapters)
        ], 1)
        tokens = self.input_norm(tokens)
        for block in self.encoder:
            tokens = block(tokens)
        return tokens, self.value_head(tokens).squeeze(-1)

    def decode(self, shifted_actions, representation):
        tokens = self.action_embedding(shifted_actions)
        for block in self.decoder:
            tokens = block(tokens, representation, self.causal_mask)
        return self.mean_head(tokens)

    def std(self):
        return (0.5 * self.std_parameter.sigmoid()).clamp_min(1e-6)

    def padded_actions(self, actions):
        return torch.stack([
            F.pad(actions[:, sl], (0, self.maximum_action_dim - dim))
            for sl, dim in zip(self.ac, self.ad)
        ], 1)

    def values(self, batch):
        _, normalized = self.encode(batch)
        return normalized, self.normalizer.denormalize(normalized)

    def evaluate_actions(self, batch):
        representation, normalized = self.encode(batch)
        actions = self.padded_actions(batch["a"])
        shifted = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), 1)
        distribution = Normal(self.decode(shifted, representation), self.std())
        logp = distribution.log_prob(actions).masked_fill(~self.action_mask, 0)
        entropy = distribution.entropy().masked_fill(~self.action_mask, 0)
        return logp, entropy, normalized

    def act(self, batch, deterministic=False):
        representation, normalized = self.encode(batch)
        shape = (len(batch["o"]), self.n, self.maximum_action_dim)
        shifted = representation.new_zeros(shape)
        padded = representation.new_zeros(shape)
        logp = representation.new_zeros(shape)
        for i, dim in enumerate(self.ad):
            mean = self.decode(shifted, representation)[:, i, :dim]
            distribution = Normal(mean, self.std()[:dim])
            action = mean if deterministic else distribution.sample()
            padded[:, i, :dim] = action
            logp[:, i, :dim] = distribution.log_prob(action)
            if i + 1 < self.n:
                shifted[:, i + 1, :dim] = action
        return {
            "a": torch.cat([padded[:, i, :d] for i, d in enumerate(self.ad)], -1),
            "lp": logp, "vn": normalized,
            "v": self.normalizer.denormalize(normalized),
        }

    def learn(self, batch):
        advantages = batch["adv"]
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-5)
        args = self.args
        stats = {}
        for indices in self.minibatches(len(batch["a"])):
            minibatch = {k: v[indices] for k, v in batch.items()}
            logp, entropy, values = self.evaluate_actions(minibatch)
            ratio = (logp - minibatch["lp"]).exp()
            adv = advantages[indices, :, None]
            objective = torch.minimum(
                ratio * adv,
                ratio.clamp(1 - args.clip_param, 1 + args.clip_param) * adv,
            ).masked_fill(~self.action_mask, 0)
            policy_loss = -objective.sum(-1).mean()
            critic_loss = value_loss(
                values, minibatch["vn"], minibatch["ret"],
                self.normalizer, args.vf_clip_param)
            loss = (
                policy_loss + args.vf_loss_coeff * critic_loss
                - args.entropy_coeff * entropy.sum(-1).mean()
            )
            total, norm = optimize(
                self.optimizers["joint"], loss, self.parameters(), args.grad_clip)
            stats = {
                "policy_loss": float(policy_loss.detach()),
                "vf_loss": float(critic_loss.detach()),
                "total_loss": total, "joint_grad_norm": norm,
            }
        return stats


class FactoredQ(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.os, self.ac = model.os, model.ac
        self.utilities = nn.ModuleList(
            facmac_mlp(o + a, 1, model.options["utility_hidden_sizes"])
            for o, a in zip(model.od, model.ad)
        )
        self.mixer = FactoredMixer(
            n_agents=model.n,
            state_dim=model.sd,
            embed_dim=model.options["mixer_embed"],
            monotonic=model.options["monotonic"],
            hypernet_embed=model.options["hypernet_embed"],
        )

    def forward(self, batch, actions):
        utilities = torch.stack([
            net(torch.cat((batch["o"][:, os], actions[:, ac]), -1)).squeeze(-1)
            for net, os, ac in zip(self.utilities, self.os, self.ac)
        ], -1)
        return self.mixer(utilities, batch["s"]).unsqueeze(-1)


class TwinQ(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.networks = nn.ModuleList(
            mlp(model.sd + sum(model.ad), 1, model.args) for _ in range(2))

    def forward(self, batch, actions):
        inputs = torch.cat((batch["s"], actions), -1)
        return torch.cat([net(inputs) for net in self.networks], -1)


class OffPolicy(NativeModel):
    def __init__(self, args, meta, options):
        super().__init__(args, meta, options)
        self.sac = args.algo == "masac"
        self.actors = nn.ModuleList([
            Gaussian(o, a, args, squashed=True) if self.sac
            else nn.Sequential(
                facmac_mlp(o, a, options["actor_hidden_sizes"]), nn.Tanh()
            )
            for o, a in zip(self.od, self.ad)
        ])
        self.q = TwinQ(self) if self.sac else FactoredQ(self)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        if self.sac:
            self.log_alpha = nn.Parameter(
                torch.full((self.n,), math.log(options["alpha_init"])))
            self.register_buffer(
                "target_entropy",
                -torch.tensor(self.ad, dtype=torch.float32)
                * options["target_entropy_scale"])
        else:
            self.target_actors = copy.deepcopy(self.actors).requires_grad_(False)

    def build_optimizers(self):
        actor_lr = (
            self.args.lr if self.sac else self.options["actor_lr"]
        )
        critic_lr = (
            self.args.critic_lr if self.sac else self.options["critic_lr"]
        )
        epsilon = (
            self.args.adam_epsilon if self.sac
            else self.options["adam_epsilon"]
        )
        self.optimizers["actors"] = torch.optim.Adam(
            self.actors.parameters(), lr=actor_lr, eps=epsilon)
        self.optimizers["critic"] = torch.optim.Adam(
            self.q.parameters(), lr=critic_lr, eps=epsilon)
        if self.sac:
            self.optimizers["alpha"] = torch.optim.Adam(
                [self.log_alpha], lr=self.options["alpha_lr"],
                eps=self.args.adam_epsilon)

    def policy(self, batch, deterministic=False, target=False):
        actors = self.target_actors if target and not self.sac else self.actors
        actions, logps = [], []
        for i, actor in enumerate(actors):
            if self.sac:
                action, logp = actor.sample(self.local(batch, i), deterministic)
            else:
                action = actor(self.local(batch, i))
                logp = action.new_zeros(len(action))
            actions.append(action)
            logps.append(logp)
        return torch.cat(actions, -1), torch.stack(logps, -1)

    def act(self, batch, deterministic=False):
        actions, logp = self.policy(batch, deterministic)
        return {"a": actions, "lp": logp}

    def td_target(self, batch):
        """One-step target; true termination blocks bootstrap, truncation does not."""
        next_batch = {"o": batch["no"], "s": batch["ns"]}
        with torch.no_grad():
            next_actions, next_logp = self.policy(next_batch, target=True)
            target_q = self.target_q(next_batch, next_actions).min(-1).values
            if self.sac:
                target_q -= (self.log_alpha.exp() * next_logp).sum(-1)
            return batch["r"] + self.args.gamma * (1 - batch["term"]) * target_q

    def learn(self, batch, *, sync_stats=True):
        args, options = self.args, self.options
        target = self.td_target(batch)
        predictions = self.q(batch, batch["a"])
        critic_loss = (predictions - target[:, None]).square().mean()
        qloss, qnorm = optimize(
            self.optimizers["critic"], critic_loss,
            self.q.parameters(),
            args.grad_clip if self.sac else options["grad_clip"],
            sync_stats=sync_stats)

        self.q.requires_grad_(False)
        try:
            actions, logp = self.policy(batch)
            q = self.q(batch, actions).min(-1).values
            if self.sac:
                actor_loss = (
                    (self.log_alpha.detach().exp() * logp).sum(-1) - q
                ).mean()
            else:
                actor_loss = -q.mean() + options["action_l2"] * actions.square().mean()
            ploss, pnorm = optimize(
                self.optimizers["actors"], actor_loss,
                self.actors.parameters(),
                args.grad_clip if self.sac else options["grad_clip"],
                sync_stats=sync_stats)
        finally:
            self.q.requires_grad_(True)

        stats = {
            "critic_loss": qloss, "actor_loss": ploss,
            "critic_grad_norm": qnorm, "actor_grad_norm": pnorm,
        }
        if self.sac:
            alpha_loss = -(
                self.log_alpha * (logp.detach() + self.target_entropy)
            ).sum(-1).mean()
            aloss, _ = optimize(
                self.optimizers["alpha"], alpha_loss, [self.log_alpha],
                args.grad_clip, sync_stats=sync_stats)
            alpha = self.log_alpha.detach().exp()
            if sync_stats:
                if not bool(torch.isfinite(alpha).all()):
                    raise FloatingPointError("Non-finite entropy temperature")
            else:
                _assert_finite_async(
                    alpha, "Non-finite entropy temperature"
                )
            stats["alpha_loss"] = aloss
            for i in range(self.n):
                stats[f"alpha_{i}"] = (
                    float(alpha[i]) if sync_stats else alpha[i].detach()
                )
        else:
            soft_update(self.target_actors, self.actors, options["tau"])
        soft_update(self.target_q, self.q, options["tau"])
        return stats


def _facmac_compile_mode():
    if os.environ.get("OMNIPIANO_TORCH_COMPILE", "0") != "1":
        return None
    mode = os.environ.get(
        "OMNIPIANO_TORCH_COMPILE_MODE", "reduce-overhead"
    )
    if mode not in {"default", "reduce-overhead"}:
        raise ValueError(
            "OMNIPIANO_TORCH_COMPILE_MODE must be 'default' or "
            "'reduce-overhead'"
        )
    return mode


def make_model(args, meta, options, device):
    if args.algo in ("happo", "ppo-monolithic"):
        model = PPO(args, meta, options)
    elif args.algo == "a2po":
        model = A2PO(args, meta, options)
    elif args.algo == "mat":
        model = MAT(args, meta, options)
    elif args.algo in ("facmac", "masac"):
        model = OffPolicy(args, meta, options)
    else:
        raise ValueError(f"No native implementation for {args.algo}")
    model.to(device)
    # Construct optimizers AFTER moving Parameters to their final device.
    model.build_optimizers()
    # FACMAC's factored critic launches many small CUDA kernels.  Compiling
    # only the online and target Q forwards is stable with the cluster PyTorch
    # build and keeps optimizer/checkpoint semantics unchanged.  Actors are
    # deliberately left eager because their collection/evaluation/training
    # batch shapes and grad modes otherwise cause repeated recompilation.
    # Compiling the complete ``learn`` method is also avoided: it captures
    # optimizer state and has proved brittle across PyTorch releases.  This is
    # an explicit runtime opt-in so checkpoints remain portable to eager
    # execution.
    if (
        args.algo == "facmac"
        and device.type == "cuda"
        and (compile_mode := _facmac_compile_mode()) is not None
    ):
        for module in (model.q, model.target_q):
            module.forward = torch.compile(
                module.forward,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )
        print(
            "[facmac] torch.compile enabled for Q forward graphs "
            f"(mode={compile_mode})"
        )
    return model
