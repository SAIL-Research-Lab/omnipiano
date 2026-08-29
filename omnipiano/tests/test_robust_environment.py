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
from omnipiano.configs import RobustConfig, RobustEnvConfig
from omnipiano.tasks.omni_piano_task import OmniPianoTask
from omnipiano.utils.env_unwrap import get_composer_env_from_gym
from omnipiano.utils.info_keys import InfoKeys


_PHYSICAL_ID_RE = re.compile(
    r"^OmniPiano-ClairDeLune-(?:G|CF|HP|GCFHP)-"
)
_COMPOUND_SHIFT = (
    "OmniPiano-ClairDeLune-GCFHP-Shift-G150-CF15-Y50-Z20-v0"
)
_HAND_SHIFT = "OmniPiano-ClairDeLune-HP-Shift-Y50-Z20-v0"


class _FakeBinding:
    def __init__(self, physics, name):
        self._physics = physics
        self._name = name

    @property
    def friction(self):
        return self._physics.frictions[self._name]

    @friction.setter
    def friction(self, value):
        self._physics.frictions[self._name] = np.asarray(value).copy()


class _FakePhysics:
    def __init__(self):
        self.model = types.SimpleNamespace(
            opt=types.SimpleNamespace(gravity=np.array([0.0, 0.0, -9.81]))
        )
        self.frictions = {
            "tips": np.array([[1.0, 0.005, 0.0001], [1.0, 0.005, 0.0001]]),
            "keys": np.array([[1.0, 0.005, 0.0001], [1.0, 0.005, 0.0001]]),
        }
        self.forward_calls = 0

    def bind(self, geoms):
        return _FakeBinding(self, geoms)

    def forward(self):
        self.forward_calls += 1


def _model_noise_task(environment_noise, seed=123):
    task = OmniPianoTask.__new__(OmniPianoTask)
    task.robust_config = RobustConfig(environment_noise=environment_noise)
    task._environment_noise_rng = np.random.default_rng(seed)
    task._nominal_gravity = np.array([0.0, 0.0, -9.81])
    task._tip_collision_geoms = "tips"
    task._key_collision_geoms = "keys"
    task._nominal_tip_friction = np.array(
        [[1.0, 0.005, 0.0001], [1.0, 0.005, 0.0001]]
    )
    task._nominal_key_friction = task._nominal_tip_friction.copy()
    task._environment_noise_state = {
        "gravity_noise": 0.0,
        "gravity_z": -9.81,
        "contact_friction_noise": 0.0,
        "contact_friction_sliding": 1.0,
        "hand_position_offsets": {},
        "hand_position_l2": 0.0,
        "hand_position_max_l2": 0.0,
    }
    return task


def test_environment_noise_frequency_defaults_and_validation():
    config = RobustEnvConfig()
    assert config.gravity_frequency == "episode"
    assert config.contact_friction_frequency == "episode"

    with pytest.raises(ValueError, match="gravity_frequency"):
        RobustEnvConfig(gravity_frequency="physics_substep")
    with pytest.raises(ValueError, match="contact_friction_frequency"):
        RobustEnvConfig(contact_friction_frequency="reset")


@pytest.mark.parametrize("dist", ["gaussian", "uniform"])
def test_step_model_noise_uses_reproducible_gravity_then_friction_draws(dist):
    kwargs = {
        "gravity_noise_dist": dist,
        "gravity_frequency": "step",
        "contact_friction_noise_dist": dist,
        "contact_friction_frequency": "step",
    }
    if dist == "gaussian":
        kwargs.update(
            gravity_noise_std=0.2,
            contact_friction_noise_std=0.03,
        )
    else:
        kwargs.update(
            gravity_noise_uniform_low=-0.2,
            gravity_noise_uniform_high=0.2,
            contact_friction_noise_uniform_low=-0.03,
            contact_friction_noise_uniform_high=0.03,
        )
    environment_noise = RobustEnvConfig(**kwargs)
    first = _model_noise_task(environment_noise, seed=77)
    second = _model_noise_task(environment_noise, seed=77)
    first_physics = _FakePhysics()
    second_physics = _FakePhysics()
    expected_rng = np.random.default_rng(77)

    first_values = []
    second_values = []
    expected_values = []
    for _ in range(4):
        first._apply_step_environment_noise(first_physics)
        second._apply_step_environment_noise(second_physics)
        first_values.append((
            first.environment_noise_state["gravity_noise"],
            first.environment_noise_state["contact_friction_noise"],
        ))
        second_values.append((
            second.environment_noise_state["gravity_noise"],
            second.environment_noise_state["contact_friction_noise"],
        ))
        if dist == "gaussian":
            expected_values.append((
                expected_rng.normal(0.0, 0.2),
                expected_rng.normal(0.0, 0.03),
            ))
        else:
            expected_values.append((
                expected_rng.uniform(-0.2, 0.2),
                expected_rng.uniform(-0.03, 0.03),
            ))

    np.testing.assert_allclose(first_values, second_values)
    np.testing.assert_allclose(first_values, expected_values)
    assert len({tuple(value) for value in first_values}) > 1
    assert first_physics.forward_calls == 0


def test_step_shift_is_constant_does_not_consume_rng_and_does_not_accumulate():
    environment_noise = RobustEnvConfig(
        gravity_noise_dist="shift",
        gravity_noise_shift=20.0,
        gravity_frequency="step",
        contact_friction_noise_dist="shift",
        contact_friction_noise_shift=-2.0,
        contact_friction_frequency="step",
    )
    task = _model_noise_task(environment_noise, seed=91)
    physics = _FakePhysics()
    untouched_rng = np.random.default_rng(91)

    for _ in range(3):
        task._apply_step_environment_noise(physics)
        state = task.environment_noise_state
        assert state["gravity_z"] == pytest.approx(task._MAX_GRAVITY_Z)
        assert state["gravity_noise"] == pytest.approx(
            task._MAX_GRAVITY_Z + 9.81
        )
        assert state["contact_friction_sliding"] == pytest.approx(
            task._MIN_FRICTION
        )
        assert state["contact_friction_noise"] == pytest.approx(
            task._MIN_FRICTION - 1.0
        )

    assert task._environment_noise_rng.random() == untouched_rng.random()
    assert physics.forward_calls == 0


def test_gravity_and_friction_frequencies_are_independent():
    environment_noise = RobustEnvConfig(
        gravity_noise_dist="shift",
        gravity_noise_shift=1.5,
        gravity_frequency="episode",
        contact_friction_noise_dist="shift",
        contact_friction_noise_shift=0.2,
        contact_friction_frequency="step",
    )
    task = _model_noise_task(environment_noise)
    physics = _FakePhysics()

    episode_values = task._sample_and_apply_model_noise(
        physics, frequency="episode"
    )
    assert episode_values[0] == pytest.approx(1.5)
    assert episode_values[1] is None
    assert physics.model.opt.gravity[2] == pytest.approx(-8.31)
    assert physics.frictions["keys"][:, 0] == pytest.approx([1.0, 1.0])

    task._apply_step_environment_noise(physics)
    assert physics.model.opt.gravity[2] == pytest.approx(-8.31)
    assert physics.frictions["keys"][:, 0] == pytest.approx([1.2, 1.2])


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
    # Existing episode tasks: 3 distributions * (3 G + 3 CF + 3 HP +
    # 1 compound) + 3 negative Shift controls = 33.
    # Step additions: Gaussian/Uniform * (3 G + 3 CF + 3 compound frequency
    # combinations) = 18. Shift is constant, so step aliases are omitted.
    ids = _physical_env_ids()
    assert len(ids) == 51, ids
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
