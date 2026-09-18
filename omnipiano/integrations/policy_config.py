"""Base class for policy configurations."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class PolicyConfig:
    total_env_steps: int = 5_000_000
    gamma: float = 0.8
    n_envs: int = 16
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str = "cpu"
