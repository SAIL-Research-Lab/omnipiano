"""Normal-Inverse-Gamma value model used by EPPO."""

from __future__ import annotations

import math

import torch
from torch import nn


class EvidentialValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...], eps: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(width, 4)
        self.eps = eps

    def forward(self, observation):
        raw_omega, raw_nu, raw_alpha, raw_beta = self.head(self.features(observation)).unbind(-1)
        omega = raw_omega.unsqueeze(-1)
        nu = (torch.nn.functional.softplus(raw_nu) + self.eps).unsqueeze(-1)
        alpha = (torch.nn.functional.softplus(raw_alpha) + 1.0 + self.eps).unsqueeze(-1)
        beta = (torch.nn.functional.softplus(raw_beta) + self.eps).unsqueeze(-1)
        return omega, nu, alpha, beta


def build_evidential_critic(obs_dim, hidden_sizes=(256, 256), eps=1e-6):
    from tensordict.nn import TensorDictModule

    return TensorDictModule(
        EvidentialValueNet(obs_dim, tuple(hidden_sizes), eps),
        in_keys=["observation"],
        out_keys=["state_value", "nig_nu", "nig_alpha", "nig_beta"],
    )


def nig_loss(tensordict, critic, regularization: float):
    """Student-t NLL plus the EPPO hyperprior regularizer."""
    critic(tensordict)
    target = tensordict["value_target"]
    omega = tensordict["state_value"]
    nu = tensordict["nig_nu"]
    alpha = tensordict["nig_alpha"]
    beta = tensordict["nig_beta"]
    omega_term = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * (math.log(math.pi) - torch.log(nu))
        - alpha * torch.log(omega_term)
        + (alpha + 0.5) * torch.log((target - omega).square() * nu + omega_term)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()

    log_prior = (
        -0.5 * (omega / 100.0).square()
        + 4.0 * torch.log(nu) - nu
        + 4.0 * torch.log(alpha - 1.0) - (alpha - 1.0)
        + 4.0 * torch.log(beta) - beta
    )
    prior_penalty = -log_prior.mean()
    return nll + regularization * prior_penalty, nll, prior_penalty


def value_variance(tensordict):
    return tensordict["nig_beta"] / (tensordict["nig_alpha"] - 1.0) * (
        1.0 + 1.0 / tensordict["nig_nu"]
    )


def uncertainty(tensordict):
    aleatoric = tensordict["nig_beta"] / (tensordict["nig_alpha"] - 1.0)
    epistemic = aleatoric / tensordict["nig_nu"]
    return aleatoric, epistemic
