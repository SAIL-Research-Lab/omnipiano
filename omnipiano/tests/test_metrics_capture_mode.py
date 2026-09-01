"""Regression tests for the train/eval physical-metrics capture switch."""

from __future__ import annotations

import dm_env
import numpy as np
import pytest
from dm_env import specs

from omnipiano.envs.robopianist.wrappers.evaluation import (
    MidiEvaluationWrapper,
)


class _Piano:
    n_keys = 88
    keys = ()
    activation = np.zeros(88, dtype=bool)
    sustain_activation = np.zeros(1, dtype=bool)


class _Task:
    piano = _Piano()
    control_timestep = 0.05
    hands_by_name = {}
    hand_specs = ()
    _notes = ((),)
    _sustains = (False,)


class _OneStepDmEnv(dm_env.Environment):
    task = _Task()
    physics = object()

    def reset(self):
        return dm_env.restart(np.zeros(1, dtype=np.float64))

    def step(self, action):
        return dm_env.termination(
            reward=0.0,
            observation=np.zeros(1, dtype=np.float64),
        )

    def observation_spec(self):
        return specs.Array((1,), np.float64)

    def action_spec(self):
        return specs.BoundedArray((1,), np.float64, -1.0, 1.0)


def _install_one_hand_metadata(wrapper: MidiEvaluationWrapper) -> None:
    wrapper._physics_metadata = {
        "hand_names": ("test_hand",),
        "hands_by_name": {},
        "fingertip_geom_to_hand": {},
        "hand_geom_to_hand": {},
        "key_geom_to_key": {},
        "hand_key_ranges": {"test_hand": (0, 87)},
    }


def test_disabled_physics_capture_is_unavailable_not_zero(monkeypatch) -> None:
    wrapper = MidiEvaluationWrapper(
        _OneStepDmEnv(),
        capture_physics_metrics=False,
    )

    def _must_not_run():
        raise AssertionError("physics capture ran in training mode")

    monkeypatch.setattr(wrapper, "_capture_physics_metrics", _must_not_run)
    wrapper.reset()
    _install_one_hand_metadata(wrapper)
    wrapper.step(np.zeros(1, dtype=np.float64))

    trace = wrapper.get_last_episode_trace()
    assert trace.key_contact_force is None
    assert trace.hand_power is None
    assert trace.hand_collision_force is None

    metrics = wrapper.get_musical_metrics()
    assert metrics["contact_attribution_available"] == 0.0
    assert metrics["inter_hand_collision_metrics_available"] == 0.0
    assert metrics["inter_agent_collision_metrics_available"] == 0.0
    assert metrics["hand_power_available"] == 0.0


def test_enabled_physics_capture_retains_empty_hand_axes(monkeypatch) -> None:
    wrapper = MidiEvaluationWrapper(
        _OneStepDmEnv(),
        capture_physics_metrics=True,
    )
    monkeypatch.setattr(
        wrapper,
        "_capture_physics_metrics",
        lambda: (
            np.zeros((1, 88), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        ),
    )
    wrapper.reset()
    _install_one_hand_metadata(wrapper)
    wrapper.step(np.zeros(1, dtype=np.float64))

    trace = wrapper.get_last_episode_trace()
    assert trace.key_contact_force.shape == (1, 1, 88)
    assert trace.hand_power.shape == (1, 1)
    assert trace.hand_collision_force.shape == (1, 1, 1)


def test_multiagent_capture_switch_rejects_string_boolean() -> None:
    from omnipiano.multiagent import make_parallel

    with pytest.raises(TypeError, match="must be a bool"):
        make_parallel(
            "OmniPiano-PianoSonataNo301StMov-FourHand-MA-Duet-"
            "Territorial-v0",
            metrics_capture_physics="false",
        )
