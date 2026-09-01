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

"""A wrapper for tracking episode statistics pertaining to music performance.

TODO(kevin):
- Look into `mir_eval` for metrics.
- Should sustain be a separate metric or should it just be applied to the note sequence
    as a whole?
"""

from collections import deque
from typing import Deque, Dict, List, Mapping, NamedTuple, Optional, Tuple

import dm_env
import mujoco
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support

from omnipiano.benchmark.metrics import EpisodeTrace, compute_episode_metrics


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        deque_size: int = 1,
        hand_to_agent: Optional[Mapping[str, str]] = None,
        capture_physics_metrics: bool = True,
    ) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []
        self._sustain_presses: List[np.ndarray] = []
        self._key_contact_forces: List[np.ndarray] = []
        self._hand_powers: List[np.ndarray] = []
        self._hand_collision_forces: List[np.ndarray] = []
        self._configured_hand_to_agent = dict(hand_to_agent or {})
        self._capture_physics_metrics_enabled = bool(capture_physics_metrics)
        self._physics_metadata = None
        self._last_episode_trace: Optional[EpisodeTrace] = None
        self._benchmark_metrics: Deque[Dict[str, float]] = deque(maxlen=deque_size)

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

        # Sustain metrics.
        self._sustain_precisions: Deque[float] = deque(maxlen=deque_size)
        self._sustain_recalls: Deque[float] = deque(maxlen=deque_size)
        self._sustain_f1s: Deque[float] = deque(maxlen=deque_size)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = self._environment.task.piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))
        if self._capture_physics_metrics_enabled:
            key_forces, hand_power, collision_forces = (
                self._capture_physics_metrics()
            )
            self._key_contact_forces.append(key_forces)
            self._hand_powers.append(hand_power)
            self._hand_collision_forces.append(collision_forces)

        if timestep.last():
            key_press_metrics = self._compute_key_press_metrics()
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            sustain_metrics = self._compute_sustain_metrics()
            self._sustain_precisions.append(sustain_metrics.precision)
            self._sustain_recalls.append(sustain_metrics.recall)
            self._sustain_f1s.append(sustain_metrics.f1)

            trace = self._build_episode_trace()
            metrics = compute_episode_metrics(trace)
            # Keep the exact historical sklearn frame-macro values under their
            # compatibility names. Metrics-v2.1's event/key-time fields remain
            # reward-independent and are not aliases for these six values.
            metrics.update({
                "precision": key_press_metrics.precision,
                "recall": key_press_metrics.recall,
                "f1": key_press_metrics.f1,
                "sustain_precision": sustain_metrics.precision,
                "sustain_recall": sustain_metrics.recall,
                "sustain_f1": sustain_metrics.f1,
            })
            self._last_episode_trace = trace
            self._benchmark_metrics.append(metrics)

            self._key_presses = []
            self._sustain_presses = []
            self._key_contact_forces = []
            self._hand_powers = []
            self._hand_collision_forces = []
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._key_contact_forces = []
        self._hand_powers = []
        self._hand_collision_forces = []
        self._physics_metadata = None
        return self._environment.reset()

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean precision/recall/F1 over the last `deque_size` episodes."""
        if not self._benchmark_metrics:
            raise ValueError("No episode metrics available yet.")
        all_keys = set().union(*(
            metrics.keys() for metrics in self._benchmark_metrics
        ))
        means = {}
        for key in all_keys:
            values = [
                metrics[key]
                for metrics in self._benchmark_metrics
                if key in metrics
            ]
            means[key] = float(np.mean(values))
        return means

    def get_last_episode_trace(self) -> EpisodeTrace:
        """Return the most recently completed raw trace for auditing/replay."""

        if self._last_episode_trace is None:
            raise ValueError("No episode trace available yet.")
        return self._last_episode_trace

    def _build_episode_trace(self) -> EpisodeTrace:
        task = self._environment.task
        steps = len(self._key_presses)
        target_keys = np.zeros((steps, task.piano.n_keys), dtype=bool)
        for step, notes in enumerate(task._notes[:steps]):
            target_keys[step, [note.key for note in notes]] = True
        target_sustain = np.zeros(steps, dtype=bool)
        available_sustain = np.asarray(task._sustains[:steps], dtype=bool)
        target_sustain[: len(available_sustain)] = available_sustain

        metadata = self._ensure_physics_metadata()
        hand_names = metadata["hand_names"]
        hand_to_agent = {
            hand: self._configured_hand_to_agent.get(hand, hand)
            for hand in hand_names
        }
        return EpisodeTrace(
            target_keys=target_keys,
            actual_keys=np.asarray(self._key_presses, dtype=bool),
            target_sustain=target_sustain,
            actual_sustain=np.asarray(self._sustain_presses, dtype=bool).reshape(
                steps
            ),
            control_timestep=float(task.control_timestep),
            hand_names=hand_names,
            hand_to_agent=hand_to_agent,
            hand_key_ranges=metadata["hand_key_ranges"],
            key_contact_force=(
                np.asarray(self._key_contact_forces, dtype=np.float64)
                if self._capture_physics_metrics_enabled
                else None
            ),
            hand_power=(
                np.asarray(self._hand_powers, dtype=np.float64)
                if self._capture_physics_metrics_enabled
                else None
            ),
            hand_collision_force=(
                np.asarray(self._hand_collision_forces, dtype=np.float64)
                if self._capture_physics_metrics_enabled
                else None
            ),
        )

    def _ensure_physics_metadata(self) -> dict:
        if self._physics_metadata is not None:
            return self._physics_metadata
        task = self._environment.task
        physics = self._environment.physics
        hands_by_name = getattr(task, "hands_by_name", {})
        hand_names = tuple(hands_by_name.keys())
        hand_index = {hand: index for index, hand in enumerate(hand_names)}

        fingertip_geom_to_hand = {}
        hand_geom_to_hand = {}
        for hand_name, hand in hands_by_name.items():
            index = hand_index[hand_name]
            for geom in hand.mjcf_model.find_all("geom"):
                hand_geom_to_hand[int(physics.bind(geom).element_id)] = index
            for body in hand.fingertip_bodies:
                for geom in body.find_all("geom"):
                    fingertip_geom_to_hand[
                        int(physics.bind(geom).element_id)
                    ] = index

        key_geom_to_key = {
            int(physics.bind(key.geom[0]).element_id): key_index
            for key_index, key in enumerate(task.piano.keys)
        }
        hand_key_ranges = {}
        for spec in getattr(task, "hand_specs", ()):
            hand_key_ranges[spec.name] = (
                spec.key_range or (0, task.piano.n_keys - 1)
            )

        self._physics_metadata = {
            "hand_names": hand_names,
            "hands_by_name": hands_by_name,
            "fingertip_geom_to_hand": fingertip_geom_to_hand,
            "hand_geom_to_hand": hand_geom_to_hand,
            "key_geom_to_key": key_geom_to_key,
            "hand_key_ranges": hand_key_ranges,
        }
        return self._physics_metadata

    def _capture_physics_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read contact forces, actuator power, and inter-hand collisions."""

        metadata = self._ensure_physics_metadata()
        physics = self._environment.physics
        task = self._environment.task
        hand_names = metadata["hand_names"]
        n_hands = len(hand_names)
        key_force = np.zeros((n_hands, task.piano.n_keys), dtype=np.float64)
        collision_force = np.zeros((n_hands, n_hands), dtype=np.float64)
        hand_power = np.zeros(n_hands, dtype=np.float64)
        for index, hand_name in enumerate(hand_names):
            hand = metadata["hands_by_name"][hand_name]
            hand_power[index] = float(
                np.sum(hand.observables.actuators_power(physics))
            )

        fingertip_map = metadata["fingertip_geom_to_hand"]
        hand_geom_map = metadata["hand_geom_to_hand"]
        key_map = metadata["key_geom_to_key"]
        for contact_index in range(int(physics.data.ncon)):
            contact = physics.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                physics.model.ptr,
                physics.data.ptr,
                contact_index,
                force6,
            )
            normal_force = abs(float(force6[0]))

            if geom1 in fingertip_map and geom2 in key_map:
                key_force[fingertip_map[geom1], key_map[geom2]] += normal_force
            elif geom2 in fingertip_map and geom1 in key_map:
                key_force[fingertip_map[geom2], key_map[geom1]] += normal_force

            hand1 = hand_geom_map.get(geom1)
            hand2 = hand_geom_map.get(geom2)
            if hand1 is not None and hand2 is not None and hand1 != hand2:
                lo, hi = sorted((hand1, hand2))
                collision_force[lo, hi] += normal_force
                collision_force[hi, lo] += normal_force
        return key_force, hand_power, collision_force

    # Helper methods.

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for key presses over the episode."""
        # Get the ground truth key presses.
        note_seq = self._environment.task._notes
        ground_truth = []
        for notes in note_seq:
            presses = np.zeros((self._environment.task.piano.n_keys,), dtype=np.float64)
            keys = [note.key for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # Deal with the case where the episode gets truncated due to a failure. In this
        # case, the length of the key presses will be less than or equal to the length
        # of the ground truth.
        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._key_presses)]

        assert len(ground_truth) == len(self._key_presses)

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._key_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for sustain presses over the episode."""
        # Get the ground truth sustain presses.
        ground_truth = [
            np.atleast_1d(v).astype(float) for v in self._environment.task._sustains
        ]

        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._sustain_presses)]

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._sustain_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)
