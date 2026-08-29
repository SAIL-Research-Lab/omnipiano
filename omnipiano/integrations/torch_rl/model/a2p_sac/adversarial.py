"""Action mixing and adaptive perturbation state for A2P-SAC."""

from __future__ import annotations

import torch
from torch import nn
from tensordict import TensorDict


class AdaptiveCoefficient:
    def __init__(
        self,
        value=0.1,
        adaptive=True,
        momentum=0.5,
        learning_rate=0.01,
        minimum=0.03,
        maximum=0.2,
        warmup_steps=5_000,
    ):
        self.value = float(value)
        self.adaptive = adaptive
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.minimum = minimum
        self.maximum = maximum
        self.moving_distance = 0.0
        self.warmup_steps = warmup_steps
        self.samples_seen = 0
        self.initialized = False

    def update(self, agent_action, adversary_action):
        distance = torch.linalg.vector_norm(agent_action - adversary_action, dim=-1).mean().item()
        sample_count = agent_action.numel() // agent_action.shape[-1]
        if not self.initialized:
            self.moving_distance = distance
            self.initialized = True
        elif self.adaptive and self.samples_seen >= self.warmup_steps:
            difference = distance - self.moving_distance
            direction = float(difference > 0.0) - float(difference < 0.0)
            change = self.learning_rate * direction * torch.sigmoid(
                torch.tensor(abs(difference))
            ).item()
            self.value = min(max(self.value - change, self.minimum), self.maximum)
        if self.initialized:
            self.moving_distance = (
                (1.0 - self.momentum) * self.moving_distance + self.momentum * distance
            )
        self.samples_seen += sample_count
        return distance


def mix_actions(agent_action, adversary_action, epsilon):
    return (1.0 - epsilon) * agent_action + epsilon * adversary_action


class A2PCollectionPolicy(nn.Module):
    def __init__(self, actor, adversary, coefficient: AdaptiveCoefficient):
        super().__init__()
        self.actor = actor
        self.adversary = adversary
        self.coefficient = coefficient

    def forward(self, tensordict):
        self.actor(tensordict)
        adversary_td = TensorDict(
            {"observation": tensordict["observation"]}, batch_size=tensordict.batch_size
        )
        self.adversary(adversary_td)
        agent_action = tensordict["action"]
        adversary_action = adversary_td["action"]
        self.coefficient.update(agent_action.detach(), adversary_action.detach())
        executed_action = mix_actions(agent_action, adversary_action, self.coefficient.value)
        tensordict["agent_action"] = agent_action
        tensordict["adversary_action"] = adversary_action
        tensordict["executed_delta"] = executed_action - agent_action
        tensordict["epsilon"] = torch.full_like(agent_action[..., :1], self.coefficient.value)
        tensordict["action"] = executed_action
        return tensordict
