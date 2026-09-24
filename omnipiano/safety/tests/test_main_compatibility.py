"""The safety evaluator must opt in to main's evaluation-only F1 wrapper."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


def test_cmdp_mode_is_scoped(monkeypatch):
    pytest.importorskip("omnisafe")
    import omnipiano
    from omnipiano.safety.cmdp import SafetySemanticCMDP, evaluation_mode
    from omnipiano.safety.suite import MAIN

    modes = []
    def make(env_id, *, mode):
        modes.append(mode)
        space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        return SimpleNamespace(observation_space=space, action_space=space)
    monkeypatch.setattr(omnipiano, "make", make)
    SafetySemanticCMDP(MAIN[0].env_id)
    with evaluation_mode():
        SafetySemanticCMDP(MAIN[0].env_id)
    SafetySemanticCMDP(MAIN[0].env_id)
    with pytest.raises(RuntimeError):
        with evaluation_mode():
            raise RuntimeError("load failure")
    SafetySemanticCMDP(MAIN[0].env_id)
    SafetySemanticCMDP(MAIN[0].env_id, mode="eval")
    assert modes == ["train", "eval", "train", "train", "eval"]
