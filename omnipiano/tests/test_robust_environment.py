"""Regression gates for episode-level physical-environment robustness."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import types

import numpy as np
import pytest

from omnipiano.envs import registration
from omnipiano.utils.env_unwrap import get_composer_env_from_gym
from omnipiano.utils.info_keys import InfoKeys


_PHYSICAL_ID_RE = re.compile(
    r"^OmniPiano-ClairDeLune-(?:G|CF|HP|GCFHP)-"
)
_COMPOUND_SHIFT = (
    "OmniPiano-ClairDeLune-GCFHP-Shift-G150-CF15-Y50-Z20-v0"
)
_HAND_SHIFT = "OmniPiano-ClairDeLune-HP-Shift-Y50-Z20-v0"


def _physical_env_ids():
    return sorted(
        env_id for env_id in registration._registry
        if _PHYSICAL_ID_RE.match(env_id)
    )


def _hand_positions(env):
    composer_env = get_composer_env_from_gym(env)
    physics = composer_env.physics
    return {
        spec.name: physics.bind(hand.root_body).xpos.copy()
        for hand, spec in zip(
            composer_env.task.hands,
            composer_env.task.hand_specs,
        )
    }


def test_physical_environment_registration_matrix():
    # 3 distributions * (3 G + 3 CF + 3 HP + 1 compound) + 3 negative
    # directional Shift controls.
    ids = _physical_env_ids()
    assert len(ids) == 33, ids
    for env_id in ids:
        config = registration._registry[env_id].robust_config
        assert config is not None
        assert any(
            config.is_channel_active(parameter)
            for parameter in config.environment_noise.PARAMETERS
        ), env_id
        assert not any(
            config.is_channel_active(channel)
            for channel in config.SIGNAL_CHANNELS
        ), env_id


def test_hand_position_shift_does_not_accumulate_between_episodes():
    env = registration.make(_HAND_SHIFT, seed=11)
    try:
        env.reset()
        first = _hand_positions(env)
        env.reset()
        second = _hand_positions(env)
        env.reset()
        third = _hand_positions(env)
    finally:
        env.close()

    for name in first:
        np.testing.assert_allclose(second[name], first[name], atol=1e-12)
        np.testing.assert_allclose(third[name], first[name], atol=1e-12)


@pytest.mark.parametrize(
    ("scale", "gravity_noise", "friction_noise", "hand_y", "hand_z"),
    [
        (0.5, 0.75, 0.075, 0.025, 0.010),
        (1.0, 1.50, 0.150, 0.050, 0.020),
    ],
)
def test_eval_scale_reaches_physical_model_and_info(
    scale, gravity_noise, friction_noise, hand_y, hand_z
):
    env = registration.make(
        _COMPOUND_SHIFT,
        mode="eval",
        eval_noise_scale=scale,
        seed=5,
    )
    try:
        env.reset(seed=5)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _, _, _, _, info = env.step(action)
    finally:
        env.close()

    assert info[InfoKeys.ROBUST_NOISE_GRAVITY] == pytest.approx(gravity_noise)
    assert info[InfoKeys.ROBUST_NOISE_CONTACT_FRICTION] == pytest.approx(
        friction_noise
    )
    assert info[InfoKeys.ROBUST_ENV_GRAVITY_Z] == pytest.approx(
        -9.81 + gravity_noise
    )
    assert info[InfoKeys.ROBUST_ENV_CONTACT_FRICTION_SLIDING] == pytest.approx(
        1.0 + friction_noise
    )
    offsets = info[InfoKeys.ROBUST_ENV_HAND_POSITION_OFFSETS]
    for offset in offsets.values():
        assert offset["y"] == pytest.approx(hand_y)
        assert offset["z"] == pytest.approx(hand_z)


def test_scale_zero_logs_nominal_absolute_physical_values():
    env = registration.make(
        _COMPOUND_SHIFT,
        mode="eval",
        eval_noise_scale=0.0,
        seed=5,
    )
    try:
        env.reset(seed=5)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _, _, _, _, info = env.step(action)
    finally:
        env.close()

    assert info[InfoKeys.ROBUST_NOISE_GRAVITY] == 0.0
    assert info[InfoKeys.ROBUST_NOISE_CONTACT_FRICTION] == 0.0
    assert info[InfoKeys.ROBUST_ENV_GRAVITY_Z] == pytest.approx(-9.81)
    assert info[InfoKeys.ROBUST_ENV_CONTACT_FRICTION_SLIDING] == pytest.approx(
        1.0
    )
    offsets = info[InfoKeys.ROBUST_ENV_HAND_POSITION_OFFSETS]
    assert offsets
    assert all(offset == {"y": 0.0, "z": 0.0} for offset in offsets.values())


def test_checkpoint_replay_rejects_physical_noise(
    tmp_path, monkeypatch
):
    # The script's guard is pure, but its module imports optional OmniSafe
    # dependencies for the CLI path. Stub those imports so this regression can
    # run in the standard SB3/pytest environment too.
    monkeypatch.setitem(sys.modules, "omnisafe", types.ModuleType("omnisafe"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(
        sys.modules,
        "run_omnisafe_template",
        types.ModuleType("run_omnisafe_template"),
    )
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "examples",
        "checkpoint_replay_eval.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_checkpoint_replay_env_robust_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "config.json").write_text(
        json.dumps({"env_id": _COMPOUND_SHIFT})
    )
    with pytest.raises(ValueError, match="gravity.*contact_friction"):
        module._assert_no_active_noise(str(tmp_path))
