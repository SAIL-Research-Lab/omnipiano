"""Small MLP modules used by the Phase-0 PPO and SAC baselines."""

from __future__ import annotations

import torch
from torch import nn


class _QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, observation, action):
        return self.net(torch.cat((observation, action), dim=-1))


def _mlp(in_features: int, hidden_sizes: tuple[int, ...], out_features: int):
    layers: list[nn.Module] = []
    width = in_features
    for hidden in hidden_sizes:
        layers += [nn.Linear(width, hidden), nn.Tanh()]
        width = hidden
    layers.append(nn.Linear(width, out_features))
    return nn.Sequential(*layers)


def build_actor(obs_dim, action_dim, action_spec, hidden_sizes=(256, 256)):
    from tensordict.nn import TensorDictModule
    from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal

    params = nn.Sequential(
        _mlp(obs_dim, tuple(hidden_sizes), 2 * action_dim),
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
    )
    module = TensorDictModule(params, in_keys=["observation"], out_keys=["loc", "scale"])
    return ProbabilisticActor(
        module=module,
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


def build_ppo_modules(obs_dim, action_dim, action_spec, hidden_sizes=(256, 256)):
    from torchrl.modules import ValueOperator

    actor = build_actor(obs_dim, action_dim, action_spec, hidden_sizes)
    critic = ValueOperator(
        _mlp(obs_dim, tuple(hidden_sizes), 1),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic


def build_sac_modules(obs_dim, action_dim, action_spec, hidden_sizes=(256, 256)):
    from tensordict.nn import TensorDictModule

    actor = build_actor(obs_dim, action_dim, action_spec, hidden_sizes)
    qvalue = TensorDictModule(
        _QNetwork(obs_dim, action_dim, tuple(hidden_sizes)),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )
    return actor, qvalue
