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


# Canonical 3-hand partition: 29/30/29 keys, middle bucket centered at y=0.
_THREE_HAND_BUCKETS = [(0, 28), (29, 58), (59, 87)]


def _three_hand_specs():
    """3-hand prototype layout — matches ``default_three_hand_specs``
    and the Task 10 registration positions. Positions = bucket centers
    of the canonical 29/30/29-key partition (N-hand morphology axiom).
    """
    return (
        HandSpec(name="rh", side=HandSide.RIGHT, position=(0.4, +0.4056, 0.13), group="treble"),
        HandSpec(name="lh", side=HandSide.LEFT, position=(0.4, -0.4051, 0.13), group="bass"),
        HandSpec(name="rh_c", side=HandSide.RIGHT, position=(0.4, 0.0, 0.13), group="middle"),
    )


def _three_hand_partition_specs():
    """3-hand layout with key_range partition matching Task 10b."""
    return (
        HandSpec(name="rh", side=HandSide.RIGHT, position=(0.4, +0.4056, 0.13),
                 key_range=(59, 87), group="treble"),
        HandSpec(name="lh", side=HandSide.LEFT, position=(0.4, -0.4051, 0.13),
                 key_range=(0, 28), group="bass"),
        HandSpec(name="rh_c", side=HandSide.RIGHT, position=(0.4, 0.0, 0.13),
                 key_range=(29, 58), group="middle"),
    )


def _get_three_hand_partition_env(control_timestep: float = 0.01):
    seq = music_pb2.NoteSequence()
    seq.notes.add(start_time=0.0, end_time=300 * control_timestep, velocity=60,
                  pitch=midi_file.note_name_to_midi_number("C4"), part=1)
    seq.total_time = 300 * control_timestep
    seq.tempos.add(qpm=60)
    midi = midi_file.MidiFile(seq=seq)
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=midi, n_steps_lookahead=0, control_timestep=control_timestep,
        change_color_on_activation=True,
        disable_fingering_reward=True,
        hand_specs=_three_hand_partition_specs(),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


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


class ThreeHandLayoutTest(absltest.TestCase):
    """Regression tests for ``default_three_hand_specs`` geometry.

    Guards against drift in the bucket-center attach positions and
    against name/side mismatches.
    """

    def test_default_three_hand_names_match_sides(self):
        """name prefix `lh_` ⇔ HandSide.LEFT, `rh_` ⇔ HandSide.RIGHT."""
        from OmniPiano.tasks.hand_spec import default_three_hand_specs
        for spec in default_three_hand_specs():
            if spec.name.startswith("lh"):
                self.assertEqual(spec.side, HandSide.LEFT,
                    msg=f"{spec.name!r} starts with 'lh' but side is {spec.side}")
            elif spec.name.startswith("rh"):
                self.assertEqual(spec.side, HandSide.RIGHT,
                    msg=f"{spec.name!r} starts with 'rh' but side is {spec.side}")

    def test_default_three_hand_unique_names(self):
        """All 3 names must be unique (spec list goes into a dict)."""
        from OmniPiano.tasks.hand_spec import default_three_hand_specs
        names = [s.name for s in default_three_hand_specs()]
        self.assertEqual(len(set(names)), len(names),
                         msg=f"duplicate hand names in 3-hand spec: {names}")

    def test_default_three_hand_spatial_layout(self):
        """Sorted by Y position: lh (bass) → rh_c (middle) → rh (treble),
        with sides [LEFT, RIGHT, RIGHT]."""
        from OmniPiano.tasks.hand_spec import default_three_hand_specs
        specs = default_three_hand_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        names = [s.name for s in ordered]
        sides = [s.side for s in ordered]
        self.assertEqual(names, ["lh", "rh_c", "rh"],
            msg=f"spatial order mismatch: {names}")
        self.assertEqual(sides,
            [HandSide.LEFT, HandSide.RIGHT, HandSide.RIGHT])

    def test_default_three_hand_center_at_origin(self):
        """The center hand (rh_c) must be exactly at y=0 — under the
        29/30/29 partition the middle bucket is centered at the keyboard
        midpoint."""
        from OmniPiano.tasks.hand_spec import default_three_hand_specs
        for spec in default_three_hand_specs():
            if spec.name == "rh_c":
                self.assertAlmostEqual(spec.position[1], 0.0, places=6,
                    msg=f"rh_c y={spec.position[1]} must be 0 under 29/30/29")


class ThreeHandPartitionPositionAlignmentTest(absltest.TestCase):
    """Verifies position/key_range invariants for 3-hand static partition.

    Mirrors FourHand/FiveHandPartitionPositionAlignmentTest. Same six
    invariants applied to the 29/30/29 partition.
    """

    def test_position_matches_bucket_center(self):
        """Each spec.position[1] is within 1 cm of its key_range bucket center."""
        from OmniPiano.tasks.hand_spec import key_index_to_y
        for spec in _three_hand_partition_specs():
            lo, hi = spec.key_range
            bucket_center = 0.5 * (key_index_to_y(lo) + key_index_to_y(hi))
            offset = abs(spec.position[1] - bucket_center)
            self.assertLess(offset, 0.01,
                msg=f"{spec.name}: position y={spec.position[1]} differs from "
                    f"bucket center {bucket_center:.4f} by {offset:.4f} m "
                    f"(>= 0.01); forearm range will be biased to one side.")

    def test_forearm_range_symmetric(self):
        """Forearm_tx joint range (post-partition) should have asymmetry < 0.05 m
        around the attach point."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        for spec in _three_hand_partition_specs():
            y_lo, y_hi = key_range_to_y_range(*spec.key_range)
            j_lo = y_lo - spec.position[1]
            j_hi = y_hi - spec.position[1]
            asymm = abs(j_lo + j_hi)
            self.assertLess(asymm, 0.05,
                msg=f"{spec.name}: forearm range [{j_lo:+.4f}, {j_hi:+.4f}] "
                    f"is asymmetric by {asymm:.4f} m (>= 0.05)")

    def test_partitions_no_overlap(self):
        """key_ranges of adjacent hands must not overlap."""
        specs = _three_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            prev_hi = prev.key_range[1]
            nxt_lo = nxt.key_range[0]
            self.assertLess(prev_hi, nxt_lo,
                msg=f"{prev.name} key_range {prev.key_range} overlaps with "
                    f"{nxt.name} key_range {nxt.key_range}")

    def test_partitions_cover_full_keyboard(self):
        """Union of all 3 key_ranges must exactly cover keys 0..87."""
        covered = set()
        for spec in _three_hand_partition_specs():
            lo, hi = spec.key_range
            covered.update(range(lo, hi + 1))
        self.assertEqual(covered, set(range(88)),
            msg=f"keys not covered: {sorted(set(range(88)) - covered)}")

    def test_attach_positions_minimum_spacing(self):
        """Adjacent hand attach positions must be ≥ 0.12 m apart."""
        specs = _three_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            spacing = nxt.position[1] - prev.position[1]
            self.assertGreater(spacing, 0.12,
                msg=f"{prev.name} (y={prev.position[1]:+.3f}) and "
                    f"{nxt.name} (y={nxt.position[1]:+.3f}) are only "
                    f"{spacing:.3f} m apart")

    def test_attach_positions_within_keyboard(self):
        """Outermost attach positions must be within keyboard half-width."""
        from robopianist.models.piano import piano_constants as pc
        half_kb = pc.PIANO_LENGTH * 0.5
        for spec in _three_hand_partition_specs():
            self.assertLess(abs(spec.position[1]), half_kb,
                msg=f"{spec.name} y={spec.position[1]} outside keyboard "
                    f"half-width ±{half_kb:.4f}")


class ThreeHandStaticPartitionTest(absltest.TestCase):
    """End-to-end tests for the 3-hand Level-1 partition env.

    Mirrors FiveHandStaticPartitionTest with 3 hands.
    """

    def test_forearm_range_overridden_in_compiled_model(self):
        """Each hand's forearm_tx joint range matches its assigned bucket."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        env = _get_three_hand_partition_env()
        env.reset()
        # env.task.hands is in spec order — matches _three_hand_partition_specs:
        # rh (treble 59-87), lh (bass 0-28), rh_c (middle 29-58).
        expected_keys_in_spec_order = [(59, 87), (0, 28), (29, 58)]
        for hand, krng in zip(env.task.hands, expected_keys_in_spec_order):
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            compiled_lo, compiled_hi = env.physics.model.jnt_range[j_id]
            pos_y = float(hand.root_body.pos[1])
            world_lo = pos_y + compiled_lo
            world_hi = pos_y + compiled_hi
            expected_lo, expected_hi = key_range_to_y_range(*krng)
            self.assertAlmostEqual(world_lo, expected_lo, places=9,
                msg=f"{hand.name} world_lo mismatch")
            self.assertAlmostEqual(world_hi, expected_hi, places=9,
                msg=f"{hand.name} world_hi mismatch")

    def test_forearm_actuator_ctrlrange_synced(self):
        """Compiled MuJoCo actuator_ctrlrange must equal compiled jnt_range
        on the forearm_tx slider — otherwise an action driving the actuator
        past the joint limit produces physically inconsistent target positions.

        Checks the COMPILED model (env.physics.model.*), not the MJCF
        objects, because the MJCF compiler is the source of truth for
        what physics actually enforces."""
        env = _get_three_hand_partition_env()
        env.reset()
        for hand in env.task.hands:
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            a = hand.mjcf_model.find('actuator', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            a_id = env.physics.bind(a).element_id
            j_lo, j_hi = env.physics.model.jnt_range[j_id]
            c_lo, c_hi = env.physics.model.actuator_ctrlrange[a_id]
            self.assertAlmostEqual(c_lo, j_lo, places=9,
                msg=f"{hand.name} compiled ctrlrange.lo {c_lo} != jnt_range.lo {j_lo}")
            self.assertAlmostEqual(c_hi, j_hi, places=9,
                msg=f"{hand.name} compiled ctrlrange.hi {c_hi} != jnt_range.hi {j_hi}")

    # NOTE: test_physics_clamps_qpos_under_extreme_ctrl deliberately omitted
    # here — the MuJoCo joint-limit enforcement is universal across
    # morphologies, so running the heavy 300-step ctrl-injection regression
    # on the 5-hand env (and historically the 4-hand env) is sufficient.

    def test_action_dim_matches_3_hands_plus_sustain(self):
        """3 hands × 22 Shadow Hand actuators each + 1 sustain = 67 action dims.
        The 22 per-hand size is the upstream Shadow Hand contract; 67 is
        the public benchmark contract for the 3-hand StaticPartition env."""
        env = _get_three_hand_partition_env()
        env.reset()
        sizes = [h.action_spec(env.physics).shape[0] for h in env.task.hands]
        self.assertEqual(len(set(sizes)), 1,
            msg=f"hands have different action sizes: {sizes}")
        per_hand = sizes[0]
        total = env.action_spec().shape[0]
        self.assertEqual(per_hand, 22,
            msg=f"per-hand action dim {per_hand} != 22 (Shadow Hand DoF contract)")
        self.assertEqual(total, 67,
            msg=f"total action dim {total} != 67 (3-hand benchmark contract)")
        self.assertEqual(total, 3 * per_hand + 1,
            msg=f"action dim {total} != 3 * {per_hand} + 1 (sustain)")


_FOUR_HAND_BUCKETS = [(0, 21), (22, 43), (44, 65), (66, 87)]


def _four_hand_partition_specs():
    """4-hand L-R-L-R duet-pair layout with static key_range partition.

    Positions = bucket centers of the canonical 4 × 22-key partition
    (N-hand morphology axiom). Must match
    ``default_four_hand_specs`` and the Task 13 registration positions
    in ``OmniPiano/envs/__init__.py`` exactly.
    """
    from OmniPiano.tasks.hand_spec import HandSpec
    return (
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.4521, 0.13), key_range=(0, 21),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.1527, 0.13), key_range=(22, 43),
                 group="mid_bass"),
        HandSpec(name="lh_t", side=HandSide.LEFT,
                 position=(0.4, +0.1528, 0.13), key_range=(44, 65),
                 group="mid_treble"),
        HandSpec(name="rh_t", side=HandSide.RIGHT,
                 position=(0.4, +0.4526, 0.13), key_range=(66, 87),
                 group="treble"),
    )


def _get_long_test_midi(dt: float = 0.01, n_steps: int = 300):
    seq = music_pb2.NoteSequence()
    seq.notes.add(start_time=0.0, end_time=n_steps * dt, velocity=60,
                  pitch=midi_file.note_name_to_midi_number("C4"), part=1)
    seq.total_time = n_steps * dt
    seq.tempos.add(qpm=60)
    return midi_file.MidiFile(seq=seq)


def _get_four_hand_partition_env(control_timestep: float = 0.01):
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=_get_long_test_midi(dt=control_timestep, n_steps=300),
        n_steps_lookahead=0,
        control_timestep=control_timestep,
        change_color_on_activation=True,
        disable_fingering_reward=True,
        hand_specs=_four_hand_partition_specs(),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class FourHandStaticPartitionTest(absltest.TestCase):
    """Regression tests for Level-1 static partitioning.

    Guards against silent breakage of the key_range / y_range mechanism.
    Targets three independent failure modes:
      (a) HandSpec validation accepts invalid partitions or rejects valid ones
      (b) key_range_to_y_range produces y bounds inconsistent with piano
          MJCF physics
      (c) The forearm_tx joint range is overridden in MJCF but not enforced
          by MuJoCo physics (e.g., wrong joint type, ctrlrange not synced).
    """

    def test_hand_spec_validation(self):
        """HandSpec enforces y_range XOR key_range and validates ranges."""
        from OmniPiano.tasks.hand_spec import HandSpec
        # Both fields → ValueError
        with self.assertRaises(ValueError):
            HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     y_range=(-0.1, 0.1), key_range=(40, 50))
        # key_range out of range
        with self.assertRaises(ValueError):
            HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     key_range=(-1, 50))
        with self.assertRaises(ValueError):
            HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     key_range=(0, 88))
        # key_range with lo > hi
        with self.assertRaises(ValueError):
            HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     key_range=(50, 30))
        # y_range with lo >= hi
        with self.assertRaises(ValueError):
            HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     y_range=(0.1, -0.1))
        # Valid partition spec — no error
        s = HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13),
                     key_range=(0, 21))
        self.assertIsNotNone(s.resolved_y_range)
        # No partition fields — resolved is None (default behavior preserved)
        s = HandSpec(name="x", side=HandSide.RIGHT, position=(0.4, 0, 0.13))
        self.assertIsNone(s.resolved_y_range)

    def test_key_index_to_y_matches_piano_mjcf(self):
        """key_index_to_y must match piano_mjcf physics for all 88 keys."""
        from OmniPiano.tasks.hand_spec import key_index_to_y
        from dm_control import mjcf
        from robopianist.models.piano import piano_mjcf
        root = piano_mjcf.build()
        all_bodies = root.worldbody.find_all('body')
        key_bodies = [b for b in all_bodies
                      if b.name and b.name.startswith(('white_key_', 'black_key_'))]
        sorted_keys = sorted(key_bodies, key=lambda b: int(b.name.split('_')[-1]))
        physics = mjcf.Physics.from_mjcf_model(root)
        mjcf_y = np.array([physics.bind(b).xpos[1] for b in sorted_keys])
        analytical_y = np.array([key_index_to_y(i) for i in range(88)])
        # Tolerance is machine epsilon (~1e-15); we set 1e-12 for safety.
        np.testing.assert_allclose(analytical_y, mjcf_y, atol=1e-12,
            err_msg="key_index_to_y diverged from piano_mjcf physics")

    def test_forearm_range_overridden_in_compiled_model(self):
        """Each hand's forearm_tx joint range in the compiled MuJoCo model
        must equal (key_range_to_y_range - hand position[1])."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        env = _get_four_hand_partition_env()
        env.reset()
        # Spec order matches _four_hand_partition_specs: bass → mid_bass →
        # mid_treble → treble (left-to-right in arena Y).
        for hand, key_rng in zip(env.task.hands, _FOUR_HAND_BUCKETS):
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            compiled_lo, compiled_hi = env.physics.model.jnt_range[j_id]
            pos_y = float(hand.root_body.pos[1])
            world_lo = pos_y + compiled_lo
            world_hi = pos_y + compiled_hi
            expected_lo, expected_hi = key_range_to_y_range(*key_rng)
            self.assertAlmostEqual(world_lo, expected_lo, places=9,
                msg=f"{hand.name} world_lo mismatch")
            self.assertAlmostEqual(world_hi, expected_hi, places=9,
                msg=f"{hand.name} world_hi mismatch")

    def test_forearm_actuator_ctrlrange_synced(self):
        """Compiled MuJoCo actuator_ctrlrange must equal compiled jnt_range
        on the forearm_tx slider — otherwise an action driving the actuator
        past the joint limit produces physically inconsistent target positions.

        Checks the COMPILED model (env.physics.model.*), not the MJCF
        objects, because the MJCF compiler is the source of truth for
        what physics actually enforces."""
        env = _get_four_hand_partition_env()
        env.reset()
        for hand in env.task.hands:
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            a = hand.mjcf_model.find('actuator', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            a_id = env.physics.bind(a).element_id
            j_lo, j_hi = env.physics.model.jnt_range[j_id]
            c_lo, c_hi = env.physics.model.actuator_ctrlrange[a_id]
            self.assertAlmostEqual(c_lo, j_lo, places=9,
                msg=f"{hand.name} compiled ctrlrange.lo {c_lo} != jnt_range.lo {j_lo}")
            self.assertAlmostEqual(c_hi, j_hi, places=9,
                msg=f"{hand.name} compiled ctrlrange.hi {c_hi} != jnt_range.hi {j_hi}")

    def test_physics_clamps_qpos_under_extreme_ctrl(self):
        """Inject ctrl WAY beyond ctrlrange via raw physics.data.ctrl write
        and step physics — qpos must end up exactly at the joint limit,
        proving physics-level enforcement (not just MJCF-level annotation)."""
        env = _get_four_hand_partition_env()
        for direction_value, expect_at in [(+5.0, "hi"), (-5.0, "lo")]:
            env.reset()
            for _ in range(300):
                for hand in env.task.hands:
                    a = hand.mjcf_model.find('actuator', 'forearm_tx')
                    a_id = env.physics.bind(a).element_id
                    env.physics.data.ctrl[a_id] = direction_value
                env.physics.step()
            for hand in env.task.hands:
                j = hand.mjcf_model.find('joint', 'forearm_tx')
                qpos = float(env.physics.bind(j).qpos[0])
                target = j.range[1] if expect_at == "hi" else j.range[0]
                self.assertAlmostEqual(qpos, target, places=3,
                    msg=f"{hand.name} qpos {qpos} != joint.{expect_at}={target}")

    def test_default_full_keyboard_path_unchanged(self):
        """A spec WITHOUT y_range or key_range must keep the original
        full-keyboard joint range — protects 2-hand and Level-3 baselines."""
        from OmniPiano.tasks.hand_spec import HandSpec
        specs = (
            HandSpec(name="rh", side=HandSide.RIGHT, position=(0.4, 0.15, 0.13)),
            HandSpec(name="lh", side=HandSide.LEFT, position=(0.4, -0.15, 0.13)),
        )
        task = piano_with_shadow_hands.PianoWithShadowHands(
            midi=_get_test_midi(dt=0.01), n_steps_lookahead=0,
            control_timestep=0.01,
            change_color_on_activation=True,
            disable_fingering_reward=True, hand_specs=specs,
        )
        env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
        env.reset()
        # Each hand's forearm_tx range should be ~[-piano.size[1] - pos.y,
        # +piano.size[1] - pos.y] (full keyboard reach).
        piano_half_width = float(task.piano.size[1])
        for hand, spec in zip(env.task.hands, specs):
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            expected_lo = -piano_half_width - spec.position[1]
            expected_hi = +piano_half_width - spec.position[1]
            self.assertAlmostEqual(j.range[0], expected_lo, places=9)
            self.assertAlmostEqual(j.range[1], expected_hi, places=9)


class FourHandPartitionPositionAlignmentTest(absltest.TestCase):
    """Verifies the position/key_range invariants for 4-hand static partition.

    Mirrors FiveHandPartitionPositionAlignmentTest. Pinned invariants:
      1. Each hand's attach position equals the center of its key_range
         bucket (within 0.01 m of bucket-center, per design intent).
      2. Forearm_tx joint range is approximately symmetric around the
         attach position (asymmetry < 0.05 m).
      3. Adjacent partitions have non-overlapping key indices.
      4. The union of all 4 partitions covers the full keyboard 0..87.
      5. Adjacent hand attach positions never overlap (Shadow Hand mesh
         width ~0.10 m → minimum spacing > 0.12 m).
      6. Outermost attach positions stay within the keyboard half-width.
    """

    def test_position_matches_bucket_center(self):
        """Each spec.position[1] is within 1 cm of its key_range bucket center."""
        from OmniPiano.tasks.hand_spec import key_index_to_y
        for spec in _four_hand_partition_specs():
            lo, hi = spec.key_range
            bucket_center = 0.5 * (key_index_to_y(lo) + key_index_to_y(hi))
            offset = abs(spec.position[1] - bucket_center)
            self.assertLess(offset, 0.01,
                msg=f"{spec.name}: position y={spec.position[1]} differs from "
                    f"bucket center {bucket_center:.4f} by {offset:.4f} m "
                    f"(>= 0.01); forearm range will be biased to one side.")

    def test_forearm_range_symmetric(self):
        """Forearm_tx joint range (post-partition) should have asymmetry < 0.05 m
        around the attach point. Bucket-center positions deliver ~1e-4 m."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        for spec in _four_hand_partition_specs():
            y_lo, y_hi = key_range_to_y_range(*spec.key_range)
            j_lo = y_lo - spec.position[1]
            j_hi = y_hi - spec.position[1]
            asymm = abs(j_lo + j_hi)  # 0 = perfectly symmetric
            self.assertLess(asymm, 0.05,
                msg=f"{spec.name}: forearm range [{j_lo:+.4f}, {j_hi:+.4f}] "
                    f"is asymmetric by {asymm:.4f} m (>= 0.05); rest position "
                    f"sits at the edge of the partition rather than the middle.")

    def test_partitions_no_overlap(self):
        """key_ranges of adjacent hands must not overlap."""
        specs = _four_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            prev_hi = prev.key_range[1]
            nxt_lo = nxt.key_range[0]
            self.assertLess(prev_hi, nxt_lo,
                msg=f"{prev.name} key_range {prev.key_range} overlaps with "
                    f"{nxt.name} key_range {nxt.key_range}")

    def test_partitions_cover_full_keyboard(self):
        """Union of all 4 key_ranges must exactly cover keys 0..87."""
        covered = set()
        for spec in _four_hand_partition_specs():
            lo, hi = spec.key_range
            covered.update(range(lo, hi + 1))
        self.assertEqual(covered, set(range(88)),
            msg=f"keys not covered: {sorted(set(range(88)) - covered)}")

    def test_attach_positions_minimum_spacing(self):
        """Adjacent hand attach positions must be ≥ 0.12 m apart."""
        specs = _four_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            spacing = nxt.position[1] - prev.position[1]
            self.assertGreater(spacing, 0.12,
                msg=f"{prev.name} (y={prev.position[1]:+.3f}) and "
                    f"{nxt.name} (y={nxt.position[1]:+.3f}) are only "
                    f"{spacing:.3f} m apart; Shadow Hand mesh ~0.10 m would "
                    f"intersect.")

    def test_attach_positions_within_keyboard(self):
        """Outermost attach positions must be within keyboard half-width."""
        from robopianist.models.piano import piano_constants as pc
        half_kb = pc.PIANO_LENGTH * 0.5
        for spec in _four_hand_partition_specs():
            self.assertLess(abs(spec.position[1]), half_kb,
                msg=f"{spec.name} y={spec.position[1]} outside keyboard "
                    f"half-width ±{half_kb:.4f}")


class FourHandDuetLayoutTest(absltest.TestCase):
    """Regression tests for the L-R-L-R duet-pair layout of
    ``default_four_hand_specs``.

    Guards against accidentally reverting to the old L-L-R-R "stacked-by-side"
    layout, which made the visual "outer pair" 0.90 m apart — incompatible
    with any human two-handed pairing.
    """

    def test_default_four_hand_is_lrlr_alternating(self):
        """Spatial order (sorted by position[1]) must alternate L-R-L-R."""
        from OmniPiano.tasks.hand_spec import default_four_hand_specs
        specs = default_four_hand_specs()
        self.assertEqual(len(specs), 4)
        # Sort by Y position, lowest to highest.
        ordered = sorted(specs, key=lambda s: s.position[1])
        sides = [s.side for s in ordered]
        self.assertEqual(
            sides,
            [HandSide.LEFT, HandSide.RIGHT, HandSide.LEFT, HandSide.RIGHT],
            msg=f"4-hand layout is not L-R-L-R; got "
                f"{[s.name for s in sides]} at y="
                f"{[s.position[1] for s in ordered]}",
        )

    def test_default_four_hand_pair_spacing(self):
        """Within each duet pair (Secondo and Primo), the LH and RH should
        be roughly 0.30 m apart — anatomically plausible for one human's
        hands. Across pairs (Secondo-RH to Primo-LH), spacing should also
        be roughly 0.30. Bucket-center positions yield gaps of
        ~0.2994 / 0.3055 / 0.2998 (slight non-uniformity reflects piano
        key geometry); 1 cm tolerance accommodates this.
        """
        from OmniPiano.tasks.hand_spec import default_four_hand_specs
        ordered = sorted(default_four_hand_specs(),
                         key=lambda s: s.position[1])
        # ordered: Secondo-LH, Secondo-RH, Primo-LH, Primo-RH
        # (-0.4521 → +0.4526, bucket centers).
        gaps = [
            ordered[i + 1].position[1] - ordered[i].position[1]
            for i in range(3)
        ]
        for g in gaps:
            self.assertAlmostEqual(g, 0.30, delta=0.01,
                msg=f"adjacent hand gaps should be ~0.30 m (±0.01), "
                    f"got {gaps}")

    def test_default_four_hand_names_consistent(self):
        """Hand names must indicate side consistently with their actual side
        attribute (lh_* must be LEFT-side, rh_* must be RIGHT-side)."""
        from OmniPiano.tasks.hand_spec import default_four_hand_specs
        for spec in default_four_hand_specs():
            if spec.name.startswith("lh"):
                self.assertEqual(spec.side, HandSide.LEFT,
                    msg=f"name {spec.name!r} starts with 'lh' but side is {spec.side}")
            elif spec.name.startswith("rh"):
                self.assertEqual(spec.side, HandSide.RIGHT,
                    msg=f"name {spec.name!r} starts with 'rh' but side is {spec.side}")


# ---------------------------------------------------------------------------
# 5-hand layout + partition tests
# ---------------------------------------------------------------------------

# Expected 5-bucket key partition (matches Task 14 registration).
_FIVE_HAND_BUCKETS = [(0, 17), (18, 35), (36, 52), (53, 70), (71, 87)]


def _five_hand_partition_specs():
    """5-hand L-R-L-R-R layout with key_range partition matching Task 14."""
    from OmniPiano.tasks.hand_spec import HandSpec
    return (
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.4817, 0.13), key_range=(0, 17),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.2345, 0.13), key_range=(18, 35),
                 group="low_mid"),
        HandSpec(name="lh_c", side=HandSide.LEFT,
                 position=(0.4, +0.0061, 0.13), key_range=(36, 52),
                 group="middle"),
        HandSpec(name="rh_t2", side=HandSide.RIGHT,
                 position=(0.4, +0.2468, 0.13), key_range=(53, 70),
                 group="high_mid"),
        HandSpec(name="rh_t1", side=HandSide.RIGHT,
                 position=(0.4, +0.4879, 0.13), key_range=(71, 87),
                 group="treble"),
    )


def _get_five_hand_partition_env(control_timestep: float = 0.01):
    seq = music_pb2.NoteSequence()
    seq.notes.add(start_time=0.0, end_time=300 * control_timestep, velocity=60,
                  pitch=midi_file.note_name_to_midi_number("C4"), part=1)
    seq.total_time = 300 * control_timestep
    seq.tempos.add(qpm=60)
    midi = midi_file.MidiFile(seq=seq)
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=midi, n_steps_lookahead=0, control_timestep=control_timestep,
        change_color_on_activation=True,
        disable_fingering_reward=True,
        hand_specs=_five_hand_partition_specs(),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


class FiveHandDuetLayoutTest(absltest.TestCase):
    """Regression tests for the L-R-L-R-R duet-extension layout of
    ``default_five_hand_specs``.

    Specifically guards against:
      (a) silent reversion to the older L-L-R-R-R "stacked-by-side" layout
      (b) drift in the bucket-center attach positions
      (c) name/side mismatch between the spec name prefix and HandSide
    """

    def test_default_five_hand_is_lrlrr(self):
        """Sides sorted by Y position must be [L, R, L, R, R]."""
        from OmniPiano.tasks.hand_spec import default_five_hand_specs
        specs = default_five_hand_specs()
        self.assertEqual(len(specs), 5)
        ordered = sorted(specs, key=lambda s: s.position[1])
        sides = [s.side for s in ordered]
        self.assertEqual(
            sides,
            [HandSide.LEFT, HandSide.RIGHT, HandSide.LEFT,
             HandSide.RIGHT, HandSide.RIGHT],
            msg=f"5-hand layout is not L-R-L-R-R; got {sides}",
        )

    def test_default_five_hand_names_match_sides(self):
        """name prefix `lh_` ⇔ HandSide.LEFT, `rh_` ⇔ HandSide.RIGHT."""
        from OmniPiano.tasks.hand_spec import default_five_hand_specs
        for spec in default_five_hand_specs():
            if spec.name.startswith("lh"):
                self.assertEqual(spec.side, HandSide.LEFT,
                    msg=f"{spec.name!r} starts with 'lh' but side is {spec.side}")
            elif spec.name.startswith("rh"):
                self.assertEqual(spec.side, HandSide.RIGHT,
                    msg=f"{spec.name!r} starts with 'rh' but side is {spec.side}")

    def test_default_five_hand_unique_names(self):
        """All 5 names must be unique (spec list goes into a dict)."""
        from OmniPiano.tasks.hand_spec import default_five_hand_specs
        names = [s.name for s in default_five_hand_specs()]
        self.assertEqual(len(set(names)), len(names),
                         msg=f"duplicate hand names in 5-hand spec: {names}")


class FiveHandPartitionPositionAlignmentTest(absltest.TestCase):
    """Verifies the position/key_range invariants for 5-hand static partition.

    These are the most error-prone arithmetic relationships, so they get
    their own test class with explicit numerical bounds:
      1. Each hand's attach position equals the center of its key_range
         bucket (within 0.01 m of bucket-center, per design intent).
      2. Forearm_tx joint range is approximately symmetric around the
         attach position (asymmetry < 0.05 m).
      3. Adjacent partitions have non-overlapping key indices.
      4. The union of all 5 partitions covers the full keyboard 0..87.
      5. Adjacent hand attach positions never overlap (Shadow Hand mesh
         width ~0.10 m → minimum spacing > 0.12 m).
    """

    def test_position_matches_bucket_center(self):
        """Each spec.position[1] is within 1 cm of its key_range bucket center."""
        from OmniPiano.tasks.hand_spec import key_index_to_y
        for spec in _five_hand_partition_specs():
            lo, hi = spec.key_range
            bucket_center = 0.5 * (key_index_to_y(lo) + key_index_to_y(hi))
            offset = abs(spec.position[1] - bucket_center)
            self.assertLess(offset, 0.01,
                msg=f"{spec.name}: position y={spec.position[1]} differs from "
                    f"bucket center {bucket_center:.4f} by {offset:.4f} m "
                    f"(>= 0.01); forearm range will be biased to one side.")

    def test_forearm_range_symmetric(self):
        """Forearm_tx joint range (post-partition) should have asymmetry < 0.05 m
        around the attach point."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        for spec in _five_hand_partition_specs():
            y_lo, y_hi = key_range_to_y_range(*spec.key_range)
            j_lo = y_lo - spec.position[1]
            j_hi = y_hi - spec.position[1]
            asymm = abs(j_lo + j_hi)  # 0 = perfectly symmetric
            self.assertLess(asymm, 0.05,
                msg=f"{spec.name}: forearm range [{j_lo:+.4f}, {j_hi:+.4f}] "
                    f"is asymmetric by {asymm:.4f} m (>= 0.05); rest position "
                    f"sits at the edge of the partition rather than the middle.")

    def test_partitions_no_overlap(self):
        """key_ranges of adjacent hands must not overlap."""
        specs = _five_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            prev_hi = prev.key_range[1]
            nxt_lo = nxt.key_range[0]
            self.assertLess(prev_hi, nxt_lo,
                msg=f"{prev.name} key_range {prev.key_range} overlaps with "
                    f"{nxt.name} key_range {nxt.key_range}")

    def test_partitions_cover_full_keyboard(self):
        """Union of all 5 key_ranges must exactly cover keys 0..87."""
        covered = set()
        for spec in _five_hand_partition_specs():
            lo, hi = spec.key_range
            covered.update(range(lo, hi + 1))
        self.assertEqual(covered, set(range(88)),
            msg=f"keys not covered: {sorted(set(range(88)) - covered)}; "
                f"keys covered twice: would have been caught by no_overlap test.")

    def test_attach_positions_minimum_spacing(self):
        """Adjacent hand attach positions must be ≥ 0.12 m apart (Shadow Hand
        mesh width ~0.10 m + small safety margin)."""
        specs = _five_hand_partition_specs()
        ordered = sorted(specs, key=lambda s: s.position[1])
        for prev, nxt in zip(ordered, ordered[1:]):
            spacing = nxt.position[1] - prev.position[1]
            self.assertGreater(spacing, 0.12,
                msg=f"{prev.name} (y={prev.position[1]:+.3f}) and "
                    f"{nxt.name} (y={nxt.position[1]:+.3f}) are only "
                    f"{spacing:.3f} m apart; Shadow Hand mesh ~0.10 m would "
                    f"intersect.")

    def test_attach_positions_within_keyboard(self):
        """Outermost attach positions must be within keyboard half-width."""
        from robopianist.models.piano import piano_constants as pc
        half_kb = pc.PIANO_LENGTH * 0.5  # ≈ 0.6105
        for spec in _five_hand_partition_specs():
            self.assertLess(abs(spec.position[1]), half_kb,
                msg=f"{spec.name} y={spec.position[1]} outside keyboard "
                    f"half-width ±{half_kb:.4f}")


class FiveHandStaticPartitionTest(absltest.TestCase):
    """End-to-end tests for the 5-hand Level-1 partition env.

    Mirrors FourHandStaticPartitionTest but with 5 hands and the L-R-L-R-R
    side layout. Validates that:
      (a) the compiled MuJoCo model gets the right joint ranges from
          spec.key_range → key_range_to_y_range conversion
      (b) actuator ctrlrange stays in sync with joint range (otherwise
          out-of-range actions could request invalid ctrl targets)
      (c) physics actually clamps qpos to the joint range under extreme
          ctrl injection (proves enforcement at simulator level)
    """

    def test_forearm_range_overridden_in_compiled_model(self):
        """Each hand's forearm_tx joint range matches its assigned bucket."""
        from OmniPiano.tasks.hand_spec import key_range_to_y_range
        env = _get_five_hand_partition_env()
        env.reset()
        for hand, krng in zip(env.task.hands, _FIVE_HAND_BUCKETS):
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            compiled_lo, compiled_hi = env.physics.model.jnt_range[j_id]
            pos_y = float(hand.root_body.pos[1])
            world_lo = pos_y + compiled_lo
            world_hi = pos_y + compiled_hi
            expected_lo, expected_hi = key_range_to_y_range(*krng)
            self.assertAlmostEqual(world_lo, expected_lo, places=9,
                msg=f"{hand.name} world_lo mismatch")
            self.assertAlmostEqual(world_hi, expected_hi, places=9,
                msg=f"{hand.name} world_hi mismatch")

    def test_forearm_actuator_ctrlrange_synced(self):
        """Compiled MuJoCo actuator_ctrlrange must equal compiled jnt_range
        on the forearm_tx slider. Checks the COMPILED model (the source of
        truth for physics enforcement), not MJCF objects."""
        env = _get_five_hand_partition_env()
        env.reset()
        for hand in env.task.hands:
            j = hand.mjcf_model.find('joint', 'forearm_tx')
            a = hand.mjcf_model.find('actuator', 'forearm_tx')
            j_id = env.physics.bind(j).element_id
            a_id = env.physics.bind(a).element_id
            j_lo, j_hi = env.physics.model.jnt_range[j_id]
            c_lo, c_hi = env.physics.model.actuator_ctrlrange[a_id]
            self.assertAlmostEqual(c_lo, j_lo, places=9,
                msg=f"{hand.name} compiled ctrlrange.lo {c_lo} != jnt_range.lo {j_lo}")
            self.assertAlmostEqual(c_hi, j_hi, places=9,
                msg=f"{hand.name} compiled ctrlrange.hi {c_hi} != jnt_range.hi {j_hi}")

    def test_physics_clamps_qpos_under_extreme_ctrl(self):
        """Inject ctrl WAY beyond range; qpos must end at joint limit."""
        env = _get_five_hand_partition_env()
        for direction_value, expect_at in [(+5.0, "hi"), (-5.0, "lo")]:
            env.reset()
            for _ in range(300):
                for hand in env.task.hands:
                    a = hand.mjcf_model.find('actuator', 'forearm_tx')
                    a_id = env.physics.bind(a).element_id
                    env.physics.data.ctrl[a_id] = direction_value
                env.physics.step()
            for hand in env.task.hands:
                j = hand.mjcf_model.find('joint', 'forearm_tx')
                qpos = float(env.physics.bind(j).qpos[0])
                target = j.range[1] if expect_at == "hi" else j.range[0]
                self.assertAlmostEqual(qpos, target, places=3,
                    msg=f"{hand.name} qpos {qpos} != joint.{expect_at}={target}")

    def test_action_dim_matches_5_hands_plus_sustain(self):
        """5 hands × 22 Shadow Hand actuators each + 1 sustain = 111 action dims.
        The 22 per-hand size is the upstream Shadow Hand contract; 111 is
        the public benchmark contract for the 5-hand StaticPartition env."""
        env = _get_five_hand_partition_env()
        env.reset()
        sizes = [h.action_spec(env.physics).shape[0] for h in env.task.hands]
        # All Shadow Hands have the same action size with default DoFs.
        self.assertEqual(len(set(sizes)), 1,
            msg=f"hands have different action sizes: {sizes}")
        per_hand = sizes[0]
        total = env.action_spec().shape[0]
        self.assertEqual(per_hand, 22,
            msg=f"per-hand action dim {per_hand} != 22 (Shadow Hand DoF contract)")
        self.assertEqual(total, 111,
            msg=f"total action dim {total} != 111 (5-hand benchmark contract)")
        self.assertEqual(total, 5 * per_hand + 1,
            msg=f"action dim {total} != 5 * {per_hand} + 1 (sustain)")

    def test_no_inter_hand_collision_at_rest(self):
        """At neutral qpos (before any ctrl), the 5 hands should not collide.
        Exposes any positioning error that would put hand meshes inside
        each other on initialization."""
        env = _get_five_hand_partition_env()
        env.reset()
        # Step once to settle physics from reset (gravity, etc.) without ctrl
        for _ in range(5):
            env.physics.step()
        # Check for any hand-vs-hand contact in the contact buffer
        n_contacts = env.physics.data.ncon
        hand_geoms = {gid: hand.name
                      for hand in env.task.hands
                      for gid in [env.physics.bind(g).element_id
                                  for g in hand.root_body.find_all('geom')]}
        inter_hand_contacts = 0
        for i in range(n_contacts):
            c = env.physics.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in hand_geoms and g2 in hand_geoms and hand_geoms[g1] != hand_geoms[g2]:
                inter_hand_contacts += 1
        self.assertEqual(inter_hand_contacts, 0,
            msg=f"{inter_hand_contacts} inter-hand contacts at rest "
                f"(out of {n_contacts} total contacts)")


# ---------------------------------------------------------------------------
# Registry consistency test — guards against drift between the test
# helpers (_three/_four/_five_hand_partition_specs) and the actual
# HandSpecs registered in OmniPiano/envs/__init__.py.
#
# Without this test, the helpers can silently diverge from the registered
# envs: partition tests would pass against the helper specs while users
# constructing the env via ``OmniPiano.make(env_id)`` would get a
# different layout at runtime.
# ---------------------------------------------------------------------------


def _hand_specs_equivalent(a, b) -> bool:
    """Two HandSpec instances are equivalent if name, side, position,
    quaternion, attachment_yaw, forearm_dofs, reduced_action_space,
    group, y_range, and key_range all match."""
    fields = (
        "name", "side", "position", "quaternion", "attachment_yaw",
        "forearm_dofs", "reduced_action_space", "group",
        "y_range", "key_range",
    )
    return all(getattr(a, f) == getattr(b, f) for f in fields)


def _diff_hand_specs(a, b) -> str:
    fields = (
        "name", "side", "position", "quaternion", "attachment_yaw",
        "forearm_dofs", "reduced_action_space", "group",
        "y_range", "key_range",
    )
    diffs = []
    for f in fields:
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            diffs.append(f"  {f}: helper={va!r} registry={vb!r}")
    return "\n".join(diffs) if diffs else "(no field-level diff, but != by equality)"


class RegisteredEnvHandSpecConsistencyTest(absltest.TestCase):
    """Asserts that the spec helpers in this test file match the HandSpecs
    actually registered in ``OmniPiano/envs/__init__.py``.

    Catches a class of silent bugs where a partition test is updated but
    the corresponding env registration is not (or vice versa) — without
    this guard, low-level tests pass while end users get a divergent env.
    """

    @classmethod
    def setUpClass(cls):
        # Trigger env registration as a side-effect of importing OmniPiano.envs.
        import OmniPiano.envs  # noqa: F401
        from OmniPiano.envs.registration import _registry
        cls._registry = _registry

    def _assert_env_matches_helper(self, env_id, helper_specs):
        spec = self._registry.get(env_id)
        self.assertIsNotNone(spec,
            msg=f"env_id {env_id!r} not in registry — registration order issue?")
        registered = spec.hand_specs
        self.assertIsNotNone(registered,
            msg=f"{env_id} registered without hand_specs — expected partition specs")
        self.assertEqual(
            len(registered), len(helper_specs),
            msg=f"{env_id}: hand count {len(registered)} != helper count "
                f"{len(helper_specs)}",
        )
        for i, (h, r) in enumerate(zip(helper_specs, registered)):
            self.assertTrue(
                _hand_specs_equivalent(h, r),
                msg=f"{env_id} hand[{i}] diverges from helper:\n"
                    f"{_diff_hand_specs(h, r)}",
            )

    def test_three_hand_winterwind_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-WinterWind-ThreeHand-StaticPartition-v0",
            _three_hand_partition_specs(),
        )

    def test_three_hand_pictures_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-PicturesGreatKiev-ThreeHand-StaticPartition-v0",
            _three_hand_partition_specs(),
        )

    def test_three_hand_polonaise_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-PolonaiseOp40No1-ThreeHand-StaticPartition-v0",
            _three_hand_partition_specs(),
        )

    def test_three_hand_sonata281_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-PianoSonataNo281StMov-ThreeHand-StaticPartition-v0",
            _three_hand_partition_specs(),
        )

    # Note: PianoSonataNo301StMov-ThreeHand-StaticPartition-v0 deliberately
    # not tested — dropped from the 3-hand registry (PolonaiseOp40No1
    # dominates it on every 3-hand metric at similar length).

    def test_four_hand_winterwind_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-WinterWind-FourHand-StaticPartition-v0",
            _four_hand_partition_specs(),
        )

    def test_four_hand_sonata_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-PianoSonataNo301StMov-FourHand-StaticPartition-v0",
            _four_hand_partition_specs(),
        )

    def test_four_hand_pictures_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-PicturesGreatKiev-FourHand-StaticPartition-v0",
            _four_hand_partition_specs(),
        )

    def test_five_hand_winterwind_static_partition(self):
        self._assert_env_matches_helper(
            "OmniPiano-WinterWind-FiveHand-StaticPartition-v0",
            _five_hand_partition_specs(),
        )


if __name__ == "__main__":
    absltest.main()
