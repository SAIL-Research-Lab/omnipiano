"""Native, feed-forward MARL learners.

No Ray dependency. All batches contain aligned JOINT environment transitions.
HAPPO/central PPO use raw Gaussian actions with execution-time clipping.
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

from omnipiano.multiagent.algos._facmac_mixer import FactoredMixer


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
    """Strict sequential HAPPO, or one centralized PPO policy."""
    on_policy = True

    def __init__(self, args, meta, options):
        super().__init__(args, meta, options)
        self.central = args.algo == "ppo-monolithic"
        self.nvalues = 1 if self.central else self.n
        actor_inputs = [self.sd] if self.central else self.od
        action_dims = [sum(self.ad)] if self.central else self.ad
        self.actors = nn.ModuleList(
            Gaussian(inp, dim, args) for inp, dim in zip(actor_inputs, action_dims))
        critic_dim = self.sd if self.central else self.sd + self.n
        self.critics = nn.ModuleList(
            mlp(critic_dim, 1, args) for _ in range(self.nvalues))
        self.normalizers = nn.ModuleList(
            ValueNorm(args) for _ in range(self.nvalues))
        self.register_buffer("agent_identity", torch.eye(self.n))

    def build_optimizers(self):
        for i in range(self.nvalues):
            self.optimizers[f"actor_{i}"] = torch.optim.Adam(
                self.actors[i].parameters(), lr=self.args.lr,
                eps=self.args.adam_epsilon)
            self.optimizers[f"critic_{i}"] = torch.optim.Adam(
                self.critics[i].parameters(), lr=self.args.critic_lr,
                eps=self.args.adam_epsilon)

    def actor_input(self, batch, i):
        return batch["s"] if self.central else self.local(batch, i)

    def critic_input(self, batch, i):
        if self.central:
            return batch["s"]
        identity = self.agent_identity[i].expand(len(batch["s"]), -1)
        return torch.cat((batch["s"], identity), -1)

    def values(self, batch):
        normalized = torch.stack([
            critic(self.critic_input(batch, i)).squeeze(-1)
            for i, critic in enumerate(self.critics)
        ], -1)
        values = torch.stack([
            self.normalizers[i].denormalize(normalized[:, i])
            for i in range(self.nvalues)
        ], -1)
        return normalized, values

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

    def learn(self, batch):
        args = self.args
        size = len(batch["a"])
        advantages = batch["adv"]
        advantages = (
            advantages - advantages.mean(0, keepdim=True)
        ) / (advantages.std(0, unbiased=False, keepdim=True) + 1e-5)
        order = torch.randperm(self.nvalues).tolist()
        log_factor = torch.zeros(size, device=batch["a"].device)
        stats = {"agent_update_order": order}

        for i in order:
            with torch.no_grad():
                before, _ = self.evaluate_actor(batch, i)
                before = before.clone()
                if not torch.allclose(
                    before, batch["lp"][:, i], atol=1e-4, rtol=1e-4
                ):
                    raise RuntimeError("Stale rollout or inconsistent stored-action logp")
            stats[f"factor_before_actor_{i}"] = float(log_factor.exp().mean())

            for indices in self.minibatches(size):
                minibatch = {k: v[indices] for k, v in batch.items()}
                logp, entropy = self.evaluate_actor(minibatch, i)
                ratio = (logp - minibatch["lp"][:, i]).exp()
                adv = advantages[indices, i]
                surrogate = torch.minimum(
                    ratio * adv,
                    ratio.clamp(1 - args.clip_param, 1 + args.clip_param) * adv,
                )
                loss = -(
                    log_factor[indices].detach().exp() * surrogate
                ).mean() - args.entropy_coeff * entropy.mean()
                loss_value, norm = optimize(
                    self.optimizers[f"actor_{i}"], loss,
                    self.actors[i].parameters(), args.grad_clip)
                stats[f"actor_{i}_loss"] = loss_value
                stats[f"actor_{i}_grad_norm"] = norm

            with torch.no_grad():
                after, _ = self.evaluate_actor(batch, i)
                log_factor += after - before
                if not bool(torch.isfinite(log_factor.exp()).all()):
                    raise FloatingPointError("Non-finite HAPPO compound factor")

        for i in range(self.nvalues):
            for indices in self.minibatches(size):
                minibatch = {k: v[indices] for k, v in batch.items()}
                prediction = self.critics[i](
                    self.critic_input(minibatch, i)).squeeze(-1)
                loss = args.vf_loss_coeff * value_loss(
                    prediction, minibatch["vn"][:, i], minibatch["ret"][:, i],
                    self.normalizers[i], args.vf_clip_param)
                value, norm = optimize(
                    self.optimizers[f"critic_{i}"], loss,
                    self.critics[i].parameters(), args.grad_clip)
                stats[f"critic_{i}_loss"] = value
                stats[f"critic_{i}_grad_norm"] = norm
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


def make_model(args, meta, options, device):
    if args.algo in ("happo", "ppo-monolithic"):
        model = PPO(args, meta, options)
    elif args.algo == "mat":
        model = MAT(args, meta, options)
    elif args.algo in ("facmac", "masac"):
        model = OffPolicy(args, meta, options)
    else:
        raise ValueError(f"No native implementation for {args.algo}")
    model.to(device)
    # Construct optimizers AFTER moving Parameters to their final device.
    model.build_optimizers()
    # FACMAC's 400-wide MLPs launch many small CUDA kernels.  Compiling only
    # their forward methods is stable with the cluster PyTorch build and keeps
    # optimizer/checkpoint semantics unchanged.  Compiling the complete
    # ``learn`` method is deliberately avoided: it captures optimizer state
    # and has proved brittle across PyTorch releases.  This is an explicit
    # runtime opt-in so checkpoints remain portable to eager execution.
    if (
        args.algo == "facmac"
        and device.type == "cuda"
        and os.environ.get("OMNIPIANO_TORCH_COMPILE", "0") == "1"
    ):
        modules = [
            *model.actors,
            *model.target_actors,
            model.q,
            model.target_q,
        ]
        for module in modules:
            module.forward = torch.compile(
                module.forward, mode="default", fullgraph=False
            )
        print("[facmac] torch.compile enabled for actor/Q forward graphs")
    return model
