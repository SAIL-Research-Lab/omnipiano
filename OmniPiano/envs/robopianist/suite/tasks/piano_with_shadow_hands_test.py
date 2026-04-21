# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for piano_with_shadow_hands_test.py."""

import itertools
from typing import Optional
from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized
from dm_control import composer
from mujoco_utils import spec_utils
from note_seq.protobuf import music_pb2

from OmniPiano.tasks.hand_spec import HandSpec
from robopianist.models.hands import HandSide
from robopianist.music import midi_file
from robopianist.suite.tasks import piano_with_shadow_hands


def _get_test_midi(dt: float = 0.01) -> midi_file.MidiFile:
    seq = music_pb2.NoteSequence()

    # C6 for 2 dts.
    seq.notes.add(
        start_time=0.0,
        end_time=2 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("C6"),
        part=1,  # Right hand index.
    )
    # G5 for 1 dt.
    seq.notes.add(
        start_time=2 * dt,
        end_time=3 * dt,
        velocity=80,
        pitch=midi_file.note_name_to_midi_number("G5"),
        part=0,  # Left hand thumb.
    )

    seq.total_time = 3 * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_env(
    control_timestep: float = 0.01,
    n_steps_lookahead: int = 0,
    n_seconds_lookahead: Optional[float] = None,
    wrong_press_termination: bool = False,
    disable_fingering_reward: bool = False,
) -> composer.Environment:
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=_get_test_midi(dt=control_timestep),
        n_steps_lookahead=n_steps_lookahead,
        n_seconds_lookahead=n_seconds_lookahead,
        control_timestep=control_timestep,
        wrong_press_termination=wrong_press_termination,
        change_color_on_activation=True,
        disable_fingering_reward=disable_fingering_reward,
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class PianoWithShadowHandsTest(parameterized.TestCase):
    @parameterized.parameters(True, False)
    def test_observables(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        timestep = env.reset()

        # Piano observables.
        self.assertIn("piano/state", timestep.observation)
        self.assertIn("piano/sustain_state", timestep.observation)

        # Goal observables.
        self.assertIn("goal", timestep.observation)
        if disable_fingering_reward:
            self.assertNotIn("fingering", timestep.observation)
        else:
            self.assertIn("fingering", timestep.observation)

        # Hand observables.
        for name in ["rh_shadow_hand", "lh_shadow_hand"]:
            self.assertIn(f"{name}/joints_pos", timestep.observation)
            # self.assertIn(f"{name}/position", timestep.observation)

    def test_action_spec(self) -> None:
        env = _get_env()
        rh_action_spec = env.task.right_hand.action_spec(env.physics)
        lh_action_spec = env.task.left_hand.action_spec(env.physics)
        combined_spec = spec_utils.merge_specs([rh_action_spec, lh_action_spec])
        actual_shape = env.action_spec().shape[0] - 1  # Don't include sustain pedal.
        expected_shape = combined_spec.shape[0]
        self.assertEqual(actual_shape, expected_shape)

        right_action = np.random.uniform(
            low=rh_action_spec.minimum, high=rh_action_spec.maximum
        ).astype(rh_action_spec.dtype)
        left_action = np.random.uniform(
            low=lh_action_spec.minimum, high=lh_action_spec.maximum
        ).astype(lh_action_spec.dtype)
        action = np.concatenate([right_action, left_action, [0]])
        env.task.before_step(env.physics, action, env.random_state)

        actual_rh_action = env.physics.bind(env.task.right_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_rh_action, right_action)
        actual_lh_action = env.physics.bind(env.task.left_hand.actuators).ctrl
        np.testing.assert_array_equal(actual_lh_action, left_action)

    def test_termination_and_discount(self) -> None:
        env = _get_env()
        action_spec = env.action_spec()
        env.reset()

        # With a dt of 0.01 and a 3 dt long midi, the episode should end after 4 steps.
        zero_action = np.zeros(action_spec.shape)
        for _ in range(3):
            timestep = env.step(zero_action)
            self.assertFalse(env.task.should_terminate_episode(env.physics))
            np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

        # 1 more step to terminate.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # No failure, so discount should be 1.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 1.0)

    @parameterized.parameters(itertools.product([0.01, 0.05, 0.1], [0, 0.01, 0.1, 1]))
    def test_n_seconds_lookahead(
        self, control_timestep: float, n_seconds_lookahead: float
    ) -> None:
        env = _get_env(
            control_timestep=control_timestep, n_seconds_lookahead=n_seconds_lookahead
        )

        actual_n_steps_lookahead = env.task._n_steps_lookahead
        expected_n_steps_lookahead = int(
            np.ceil(n_seconds_lookahead / control_timestep)
        )
        self.assertEqual(actual_n_steps_lookahead, expected_n_steps_lookahead)

    @parameterized.parameters(0, 1, 2, 5)
    def test_goal_observable_lookahead(self, n_steps_lookahead: int) -> None:
        env = _get_env(control_timestep=0.01, n_steps_lookahead=n_steps_lookahead)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        sustains = note_traj.sustains
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_goal = np.zeros((n_steps_lookahead + 1, env.task.piano.n_keys + 1))

            t_start = i
            t_end = min(i + n_steps_lookahead + 1, len(notes))
            for j, t in enumerate(range(t_start, t_end)):
                keys = [note.key for note in notes[t]]
                expected_goal[j, keys] = 1.0
                expected_goal[j, -1] = sustains[t]

            actual_goal = timestep.observation["goal"]
            np.testing.assert_array_equal(actual_goal, expected_goal.ravel())

            # Check that the 0th goal is always the goal at the current timestep.
            expected_current = np.zeros((env.task.piano.n_keys + 1,))
            keys = [note.key for note in notes[i]]
            expected_current[keys] = 1.0
            expected_current[-1] = sustains[i]
            actual_current = timestep.observation["goal"][0 : env.task.piano.n_keys + 1]
            np.testing.assert_array_equal(actual_current, expected_current)

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the goal for the current timestep
            # to compute the reward. Let's check that it matches the expected goal.
            np.testing.assert_array_equal(expected_current, env.task._goal_current)

    def test_fingering_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        timestep = env.reset()

        midi = _get_test_midi(dt=0.01)
        note_traj = midi_file.NoteTrajectory.from_midi(
            midi, dt=env.task.control_timestep
        )
        notes = note_traj.notes
        self.assertLen(notes, 4)

        for i in range(len(notes)):
            expected_fingering = np.zeros((2, 5))
            idxs = [note.fingering for note in notes[i]]
            rh_idxs = [idx for idx in idxs if idx < 5]
            lh_idxs = [idx - 5 for idx in idxs if idx >= 5]
            expected_fingering[0, rh_idxs] = 1.0
            expected_fingering[1, lh_idxs] = 1.0

            actual_fingering = timestep.observation["fingering"]
            np.testing.assert_array_equal(actual_fingering, expected_fingering.ravel())

            timestep = env.step(zero_action)

            # In the `after_step` method, we cache the fingering information for the
            # current timestep to compute the reward. Let's check that it matches the
            # expected one.
            actual_rh_current = [r[1] for r in env.task._rh_keys_current]
            np.testing.assert_array_equal(rh_idxs, actual_rh_current)
            actual_lh_current = [r[1] for r in env.task._lh_keys_current]
            np.testing.assert_array_equal(lh_idxs, actual_lh_current)

    def test_failure_termination(self) -> None:
        env = _get_env(wrong_press_termination=True)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        # Simulate a wrong press by applying a generalized force on all the keys.
        env.physics.bind(env.task.piano.joints).qfrc_applied = 3.0

        # The episode should terminate in a single step.
        timestep = env.step(zero_action)
        self.assertTrue(timestep.last())
        self.assertTrue(env.task.should_terminate_episode(env.physics))
        # Failure, so discount should be 0.0.
        np.testing.assert_array_equal(env.task.get_discount(env.physics), 0.0)

    @absltest.skip("this observable is disabled")
    def test_steps_left_observable(self) -> None:
        env = _get_env(control_timestep=0.01)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)

        timestep = env.reset()
        self.assertEqual(timestep.observation["steps_left"], 1.0)

        for i in range(3):
            timestep = env.step(zero_action)
            self.assertAlmostEqual(
                timestep.observation["steps_left"], 1.0 - (i + 1) / 3
            )

    @parameterized.parameters(True, False)
    def test_fingering_reward_presence(self, disable_fingering_reward: bool) -> None:
        env = _get_env(disable_fingering_reward=disable_fingering_reward)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape)
        env.reset()

        env.step(zero_action)
        reward_terms = env.task.reward_fn.reward_terms

        if disable_fingering_reward:
            self.assertNotIn("fingering_reward", reward_terms)
        else:
            self.assertIn("fingering_reward", reward_terms)

    # TODO(kevin): Add unit tests for individual reward components.
    # TODO(kevin): Add unit tests for augmentation / midi selection.


def _three_hand_specs():
    return (
        HandSpec(name="rh", side=HandSide.RIGHT, position=(0.4, 0.30, 0.13), group="treble"),
        HandSpec(name="lh", side=HandSide.LEFT, position=(0.4, -0.30, 0.13), group="bass"),
        HandSpec(name="rh_c", side=HandSide.RIGHT, position=(0.4, 0.0, 0.13), group="middle"),
    )


def _get_three_hand_env(control_timestep: float = 0.01) -> composer.Environment:
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=_get_test_midi(dt=control_timestep),
        n_steps_lookahead=0,
        control_timestep=control_timestep,
        change_color_on_activation=True,
        disable_fingering_reward=True,
        hand_specs=_three_hand_specs(),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class ThreeHandRegressionTest(absltest.TestCase):
    """Regression tests for N-hand generalization.

    These guard against reintroducing 2-hand hardcoding in reward / observable
    paths (the class of silent bug that made OT / forearm reward underreport
    before the Phase 1 N-hand refactor).
    """

    def test_hand_count_and_observables(self) -> None:
        env = _get_three_hand_env()
        timestep = env.reset()

        self.assertLen(env.task.hands, 3)
        self.assertEqual(
            list(env.task.hands_by_name.keys()), ["rh", "lh", "rh_c"]
        )
        for name in ["rh_shadow_hand", "lh_shadow_hand", "rh_c_shadow_hand"]:
            self.assertIn(f"{name}/joints_pos", timestep.observation)

    def test_before_step_offset_dispatch_heterogeneous_hands(self) -> None:
        """The offset loop in before_step must route each action slice to its
        own hand's actuators, including when hands have different action sizes.

        Regression guard: the pre-refactor `np.split(action[:-1], 2)` would
        (a) crash on 3+ hands and (b) silently misalign if any two hands had
        different action sizes. The offset loop handles both.
        """
        specs = (
            HandSpec(name="rh", side=HandSide.RIGHT, position=(0.4, 0.30, 0.13)),
            HandSpec(
                name="lh", side=HandSide.LEFT, position=(0.4, -0.30, 0.13),
                reduced_action_space=True,  # smaller spec than the other two
            ),
            HandSpec(name="rh_c", side=HandSide.RIGHT, position=(0.4, 0.0, 0.13)),
        )
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.01),
            n_steps_lookahead=0,
            control_timestep=0.01,
            change_color_on_activation=True,
            disable_fingering_reward=True,
            hand_specs=specs,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()

        sizes = [h.action_spec(env.physics).shape[0] for h in env.task.hands]
        # Heterogeneity is load-bearing: on equal sizes, a buggy equal-split
        # dispatch would also pass this test.
        self.assertNotEqual(
            sizes[0], sizes[1],
            msg="Test setup invariant broken: lh should differ in size from rh.",
        )
        self.assertEqual(sum(sizes) + 1, env.action_spec().shape[0])

        # Build action with a distinct constant per hand so any misrouted slice
        # shows up as a mismatch rather than a numerically plausible value.
        per_hand_parts = []
        expected_per_hand = {}
        for i, hand in enumerate(env.task.hands):
            spec = hand.action_spec(env.physics)
            vals = np.full(spec.shape, float(i + 1), dtype=spec.dtype)
            per_hand_parts.append(vals)
            expected_per_hand[hand.name] = vals
        sustain_val = np.array([0.7], dtype=per_hand_parts[0].dtype)
        action = np.concatenate(per_hand_parts + [sustain_val])

        # Explicit offset-loop invariant: after consuming all hand slices, the
        # cursor must land exactly at `len(action) - 1` so the remaining 1
        # byte is sustain. If sum(sizes) < len(action) - 1, some action bytes
        # are silently dropped; if > len(action) - 1, the final hand's slice
        # would overrun into the sustain byte (or past the end).
        self.assertEqual(
            sum(sizes),
            action.shape[0] - 1,
            msg="offset-loop invariant broken: sum(per-hand sizes) != len(action[:-1]).",
        )

        env.task.before_step(env.physics, action, env.random_state)

        for hand in env.task.hands:
            actual_ctrl = env.physics.bind(hand.actuators).ctrl
            np.testing.assert_array_equal(
                actual_ctrl,
                expected_per_hand[hand.name],
                err_msg=(
                    f"Hand '{hand.name}' received wrong slice "
                    "(offset loop broken)."
                ),
            )

        # Sustain goes to piano._sustain_state directly (no MuJoCo actuator),
        # so verify via the cached state.
        np.testing.assert_array_equal(
            env.task.piano._sustain_state, sustain_val,
        )

    def test_ot_cost_matrix_uses_all_fingertips(self) -> None:
        """OT cost matrix must have 5 * N_hands rows, not a hardcoded 10."""
        env = _get_three_hand_env()
        env.reset()

        import scipy.optimize
        captured = {}
        real = scipy.optimize.linear_sum_assignment

        def spy(cost_matrix, *args, **kwargs):
            captured.setdefault("shape", cost_matrix.shape)
            return real(cost_matrix, *args, **kwargs)

        zero_action = np.zeros(env.action_spec().shape)
        with mock.patch(
            "robopianist.suite.tasks.piano_with_shadow_hands.linear_sum_assignment",
            side_effect=spy,
        ):
            env.step(zero_action)

        self.assertIn("shape", captured)
        self.assertEqual(captured["shape"][0], 5 * len(env.task.hands))


if __name__ == "__main__":
    absltest.main()
