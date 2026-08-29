"""NumPy prediction and inference-checkpoint adapter for TorchRL actors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..log.checkpoint import save_checkpoint
from .networks import build_actor
from .ompo.occupancy import build_ompo_actor


ACTOR_BUILDERS = {
    "default": build_actor,
    "ompo": build_ompo_actor,
}


class TorchRLPolicyAdapter:
    def __init__(self, actor, model_spec: dict, device: str = "cpu"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.model_spec = model_spec

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        from tensordict import TensorDict
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        observation = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        td = TensorDict({"observation": observation}, batch_size=observation.shape[:-1])
        exploration = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
        with torch.no_grad(), set_exploration_type(exploration):
            action = self.actor(td)["action"]
        return action.detach().cpu().numpy()

    def save(self, path: Path) -> None:
        save_checkpoint(path, actor=self.actor, model_spec=self.model_spec)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TorchRLPolicyAdapter":
        from torchrl.data import Bounded

        path = Path(path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(path / "model.pt", map_location=device, weights_only=True)
        network = payload["model_spec"]
        action_spec = Bounded(
            low=torch.tensor(network["action_low"], dtype=torch.float32),
            high=torch.tensor(network["action_high"], dtype=torch.float32),
            shape=torch.Size([network["action_dim"]]),
            dtype=torch.float32,
        )
        actor_builder = ACTOR_BUILDERS[network.get("model_type", "default")]
        actor = actor_builder(
            network["observation_dim"],
            network["action_dim"],
            action_spec,
            tuple(network["hidden_sizes"]),
        )
        actor.load_state_dict(payload["state_dict"])
        return cls(actor, network, device)
