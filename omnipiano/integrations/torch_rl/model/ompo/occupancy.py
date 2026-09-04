"""Transition-occupancy discriminator used by OMPO."""

from __future__ import annotations

import torch
from torch import nn


def _initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.zeros_(module.bias)


class OMPOActorParams(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        first, second = hidden_sizes
        self.linear1 = nn.Linear(observation_dim, first)
        self.layer_norm = nn.LayerNorm(first)
        self.linear2 = nn.Linear(first, second)
        self.mean = nn.Linear(second, action_dim)
        self.log_std = nn.Linear(second, action_dim)
        self.apply(_initialize)

    def forward(self, observation):
        hidden = torch.tanh(self.layer_norm(self.linear1(observation)))
        hidden = torch.nn.functional.elu(self.linear2(hidden))
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0).exp().clamp_min(1e-4)


def build_ompo_actor(observation_dim, action_dim, action_spec, hidden_sizes=(256, 256)):
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor, TanhNormal

    parameters = TensorDictModule(
        OMPOActorParams(observation_dim, action_dim, tuple(hidden_sizes)),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    return ProbabilisticActor(
        module=parameters,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
        log_prob_keys=["sample_log_prob"],
    )


class TransitionDiscriminator(nn.Module):
    def __init__(self, observation_indices, action_dim, hidden_size=256):
        super().__init__()
        self.register_buffer("observation_indices", observation_indices)
        input_dim = 2 * observation_indices.numel() + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def inputs(self, tensordict):
        state = tensordict["observation"].index_select(-1, self.observation_indices)
        next_state = tensordict["next", "observation"].index_select(
            -1, self.observation_indices
        )
        return torch.cat((state, tensordict["action"], next_state), dim=-1)

    def forward(self, tensordict):
        return self.network(self.inputs(tensordict))


class TwinQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        first, second = hidden_sizes
        input_dim = observation_dim + action_dim
        self.q1_linear1 = nn.Linear(input_dim, first)
        self.q1_norm = nn.LayerNorm(first)
        self.q1_linear2 = nn.Linear(first, second)
        self.q1_output = nn.Linear(second, 1)
        self.q2_linear1 = nn.Linear(input_dim, first)
        self.q2_norm = nn.LayerNorm(first)
        self.q2_linear2 = nn.Linear(first, second)
        self.q2_output = nn.Linear(second, 1)
        self.apply(_initialize)

    def forward(self, observation, action):
        value = torch.cat((observation, action), dim=-1)
        q1 = torch.tanh(self.q1_norm(self.q1_linear1(value)))
        q1 = torch.nn.functional.elu(self.q1_linear2(q1))
        q2 = torch.tanh(self.q2_norm(self.q2_linear1(value)))
        q2 = torch.nn.functional.elu(self.q2_linear2(q2))
        return self.q1_output(q1), self.q2_output(q2)


def discriminator_loss(discriminator, local_batch, global_batch, gradient_penalty=10.0):
    local_input = discriminator.inputs(local_batch)
    global_input = discriminator.inputs(global_batch)
    local_logits = discriminator.network(local_input)
    global_logits = discriminator.network(global_input)
    classification = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            local_logits, torch.ones_like(local_logits)
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            global_logits, torch.zeros_like(global_logits)
        )
    )
    alpha = torch.rand(local_input.shape[0], 1, device=local_input.device)
    mixed = (alpha * local_input + (1.0 - alpha) * global_input).requires_grad_(True)
    mixed_logits = discriminator.network(mixed)
    gradient = torch.autograd.grad(
        mixed_logits, mixed, torch.ones_like(mixed_logits), create_graph=True
    )[0]
    penalty = gradient_penalty * (gradient.norm(2, dim=-1) - 1.0).square().mean()
    accuracy = 0.5 * (
        (local_logits > 0).float().mean() + (global_logits < 0).float().mean()
    )
    return classification + penalty, accuracy
