"""Fast unit tests for MARL coordination metrics (no MuJoCo required)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_METRICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "multiagent"
    / "coordination_metrics.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "omnipiano_coordination_metrics_test", _METRICS_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
metrics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = metrics
_SPEC.loader.exec_module(metrics)

_WANDB_PATH = Path(__file__).resolve().parents[1] / "multiagent" / "_wandb.py"
_WANDB_SPEC = importlib.util.spec_from_file_location(
    "omnipiano_coordination_wandb_test", _WANDB_PATH
)
assert _WANDB_SPEC is not None and _WANDB_SPEC.loader is not None
wandb_helpers = importlib.util.module_from_spec(_WANDB_SPEC)
_WANDB_SPEC.loader.exec_module(wandb_helpers)


def _note(key: int) -> SimpleNamespace:
    return SimpleNamespace(key=key)


def _activation(*keys: int) -> np.ndarray:
    result = np.zeros(88, dtype=bool)
    result[list(keys)] = True
    return result


class CoordinationMetricsTest(unittest.TestCase):
    def test_common_keys_are_precise_reach_intersections(self):
        self.assertEqual(
            metrics.compute_common_keys(
                {"left": (0, 45), "right": (42, 87)}
            ),
            (42, 43, 44, 45),
        )
        self.assertEqual(
            metrics.compute_common_keys(
                {"left": (0, 30), "center": (28, 60), "right": (58, 87)}
            ),
            (28, 29, 30, 58, 59, 60),
        )

    def test_event_rates_count_onsets_not_sustained_frames(self):
        # Key 42 has two target onsets.  The first is held for two frames but
        # must still contribute only one event to the denominator.
        trajectory = [
            [],
            [_note(10), _note(42)],
            [_note(10), _note(42)],
            [],
            [_note(42)],
        ]
        accumulator = metrics.CoordinationEpisodeAccumulator([42])
        accumulator.reset(trajectory)

        accumulator.observe_step(
            _activation(), {}, inter_agent_collision=False
        )
        accumulator.observe_step(
            _activation(42),
            {42: {"left", "right"}},
            inter_agent_collision=False,
        )
        accumulator.observe_step(
            _activation(42), {42: {"left"}}, inter_agent_collision=True
        )
        accumulator.observe_step(
            _activation(), {}, inter_agent_collision=False
        )
        accumulator.observe_step(
            _activation(), {}, inter_agent_collision=False
        )

        result = accumulator.finalize()
        self.assertEqual(result[metrics.COMMON_AREA_TARGET_COUNT], 2.0)
        self.assertEqual(result[metrics.COMMON_AREA_SUCCESS_COUNT], 1.0)
        self.assertEqual(
            result[metrics.COMMON_AREA_DUPLICATE_PRESS_COUNT], 1.0
        )
        self.assertEqual(result[metrics.COMMON_AREA_SUCCESS_RATE], 0.5)
        self.assertEqual(
            result[metrics.COMMON_AREA_DUPLICATE_PRESS_RATE], 0.5
        )
        self.assertEqual(
            result[metrics.INTER_AGENT_COLLISION_STEP_RATE], 1.0 / 5.0
        )

    def test_onset_window_can_accept_one_step_late_press(self):
        accumulator = metrics.CoordinationEpisodeAccumulator(
            [42], onset_window_steps=2
        )
        accumulator.reset([[_note(42)], [_note(42)]])
        accumulator.observe_step(
            _activation(), {}, inter_agent_collision=False
        )
        accumulator.observe_step(
            _activation(42), {42: {"left"}}, inter_agent_collision=False
        )
        result = accumulator.finalize()
        self.assertEqual(result[metrics.COMMON_AREA_SUCCESS_RATE], 1.0)
        self.assertEqual(
            result[metrics.COMMON_AREA_DUPLICATE_PRESS_RATE], 0.0
        )

    def test_no_common_target_is_explicitly_not_applicable(self):
        accumulator = metrics.CoordinationEpisodeAccumulator([42])
        accumulator.reset([[_note(10)]])
        accumulator.observe_step(
            _activation(10), {}, inter_agent_collision=False
        )
        result = accumulator.finalize()
        self.assertEqual(result[metrics.COMMON_AREA_TARGET_COUNT], 0.0)
        self.assertIsNone(result[metrics.COMMON_AREA_SUCCESS_RATE])
        self.assertIsNone(result[metrics.COMMON_AREA_DUPLICATE_PRESS_RATE])

    def test_contacts_are_attributed_to_agents_and_deduplicated(self):
        contacts = [
            SimpleNamespace(geom=(1, 100), dist=0.0),
            SimpleNamespace(geom=(2, 100), dist=-1e-9),
            # A separate palm/hand collision between agents.
            SimpleNamespace(geom=(3, 4), dist=0.0),
            # An inactive positive-distance contact must be ignored.
            SimpleNamespace(geom=(5, 101), dist=1e-4),
        ]
        key_agents, collision = metrics.classify_contacts(
            contacts,
            hand_geom_to_agent={1: "left", 2: "right", 3: "left", 4: "right"},
            fingertip_geom_to_agent={1: "left", 2: "right", 5: "left"},
            key_geom_to_key={100: 42, 101: 43},
            common_keys={42, 43},
        )
        self.assertEqual(key_agents, {42: {"left", "right"}})
        self.assertTrue(collision)

        # Two fingertip contacts owned by the same agent collapse to one owner
        # and an intra-agent hand contact is not an inter-agent collision.
        key_agents, collision = metrics.classify_contacts(
            [
                SimpleNamespace(geom=(1, 100), dist=0.0),
                SimpleNamespace(geom=(2, 100), dist=0.0),
                SimpleNamespace(geom=(1, 2), dist=0.0),
            ],
            hand_geom_to_agent={1: "left", 2: "left"},
            fingertip_geom_to_agent={1: "left", 2: "left"},
            key_geom_to_key={100: 42},
            common_keys={42},
        )
        self.assertEqual(key_agents, {42: {"left"}})
        self.assertFalse(collision)

    def test_wandb_eval_logs_all_three_headline_rates(self):
        class _CaptureRun:
            def __init__(self):
                self.rows = []

            def log(self, row, step):
                self.rows.append((dict(row), step))

        capture = _CaptureRun()
        run = wandb_helpers.WandbRun.__new__(wandb_helpers.WandbRun)
        run._run = capture
        run._degraded = False
        run._last_step = -1

        run.log_eval(
            50_000,
            {
                "summary": {
                    "episode_coordination/common_area_success_rate_mean": 0.8,
                    "episode_coordination/common_area_duplicate_press_rate_mean": 0.2,
                    "episode_coordination/inter_agent_collision_step_rate_mean": 0.1,
                }
            },
        )
        self.assertEqual(len(capture.rows), 1)
        row, step = capture.rows[0]
        self.assertEqual(step, 50_000)
        self.assertEqual(
            row["eval/coordination/common_area_success_rate"], 0.8
        )
        self.assertEqual(
            row["eval/coordination/common_area_duplicate_press_rate"], 0.2
        )
        self.assertEqual(
            row["eval/coordination/inter_agent_collision_step_rate"], 0.1
        )

    def test_wandb_eval_video_is_logged_immediately_at_env_step(self):
        class _CaptureRun:
            def __init__(self):
                self.rows = []

            def log(self, row, step):
                self.rows.append((dict(row), step))

        class _FakeWandb:
            @staticmethod
            def Video(path, *, format, caption):
                return {"path": path, "format": format, "caption": caption}

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "00000.mp4"
            video.write_bytes(b"fake-mp4")
            capture = _CaptureRun()
            run = wandb_helpers.WandbRun.__new__(wandb_helpers.WandbRun)
            run._run = capture
            run._wandb = _FakeWandb()
            run._degraded = False
            run._last_step = -1

            run.log_eval_videos(
                504_000, [video], scheduled_env_step=500_000
            )

            self.assertEqual(len(capture.rows), 1)
            row, step = capture.rows[0]
            self.assertEqual(step, 504_000)
            self.assertEqual(row["env_steps"], 504_000)
            self.assertEqual(row["eval/video"]["path"], str(video))
            self.assertIn("500,000", row["eval/video"]["caption"])


if __name__ == "__main__":
    unittest.main()
