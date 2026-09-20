"""Replay a frozen keyframe trajectory."""

import json
from pathlib import Path

import numpy as np

from omnipiano.utils.env_unwrap import get_composer_env_from_gym

from .policy_config import KeyframeConfig


class KeyframePolicy:
    def __init__(self, config: KeyframeConfig):
        self.env = config.env
        self.action_space = config.action_space
        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(
                f"keyframes not found: {path}; run examples/run_keyframes.py first"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.actions = np.asarray(data["actions"], dtype=np.float32)
        expected_steps = len(get_composer_env_from_gym(self.env).task._notes)
        expected = (expected_steps, self.action_space.shape[0])
        if (
            self.actions.ndim != 2
            or self.actions.shape != expected
        ):
            raise ValueError(
                f"keyframes must cover the full song with shape {expected}; "
                f"got {self.actions.shape}"
            )
        self.actions = np.clip(
            self.actions,
            self.action_space.low,
            self.action_space.high,
        )

    def predict(self, observation, deterministic=True):
        del observation, deterministic
        step = get_composer_env_from_gym(self.env).task._t_idx
        return self.actions[step], None
