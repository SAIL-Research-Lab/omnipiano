"""Fast tests for IPPO scientific bookkeeping (no Ray/MuJoCo required)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


_COMMON_PATH = (
    Path(__file__).resolve().parents[1]
    / "multiagent"
    / "training"
    / "runtime.py"
)
_SPEC = importlib.util.spec_from_file_location("omnipiano_ippo_common_test", _COMMON_PATH)
assert _SPEC is not None and _SPEC.loader is not None
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)


TERMINAL_METRICS = {
    "episode_task/musical_f1": 0.5,
    "episode_task/musical_precision": 0.6,
    "episode_task/musical_recall": 0.4,
    "episode_task/sustain_f1": 0.7,
    "episode_coordination/common_area_success_rate": 0.75,
    "episode_coordination/common_area_duplicate_press_rate": 0.25,
    "episode_coordination/inter_agent_collision_step_rate": 0.1,
}


class _Box:
    def __init__(self, shape=(1,)):
        self.shape = shape
        self.low = np.full(shape, -1.0, dtype=np.float32)
        self.high = np.full(shape, 1.0, dtype=np.float32)
        self.dtype = np.dtype(np.float32)

    def contains(self, value):
        value = np.asarray(value)
        return (
            value.shape == self.shape
            and value.dtype == self.dtype
            and np.isfinite(value).all()
            and (value >= self.low).all()
            and (value <= self.high).all()
        )


class _TwoStepSharedEnv:
    possible_agents = ["left", "right"]

    def __init__(
        self, *, include_metrics=True, unequal_rewards=False, close_raises=False
    ):
        self.include_metrics = include_metrics
        self.unequal_rewards = unequal_rewards
        self.close_raises = close_raises
        self.agents = []
        self.step_count = 0
        self.reset_seeds = []
        self.closed = False

    def action_space(self, agent):
        del agent
        return _Box()

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0
        return {
            agent: np.zeros(2, dtype=np.float32) for agent in self.agents
        }, {agent: {} for agent in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.step_count += 1
        rewards = {"left": 1.0, "right": 2.0 if self.unequal_rewards else 1.0}
        done = self.step_count == 2
        infos = {agent: {} for agent in self.agents}
        if done and self.include_metrics:
            infos["_global_"] = dict(TERMINAL_METRICS)
        old_agents = list(self.agents)
        if done:
            self.agents = []
        observations = {
            agent: np.zeros(2, dtype=np.float32) for agent in old_agents
        }
        terms = {agent: done for agent in old_agents}
        truncs = {agent: False for agent in old_agents}
        return observations, rewards, terms, truncs, infos

    def close(self):
        self.closed = True
        if self.close_raises:
            raise RuntimeError("cleanup failed")


def _zero_action(agent, observation, action_space):
    del agent, observation
    return np.zeros(action_space.shape, dtype=action_space.dtype)


class _FakeModule:
    def __init__(self, action):
        self.action = action

    def parameters(self):
        return iter(())

    def forward_inference(self, batch):
        import torch

        assert tuple(batch["obs"].shape) == (1, 2)
        return {"actions": torch.as_tensor([self.action], dtype=torch.float32)}


class _FakeAlgorithm:
    def __init__(self, action):
        self.module = _FakeModule(action)

    def get_module(self, agent_id):
        assert agent_id == "left"
        return self.module


class _FakeCheckpointAlgorithm:
    def __init__(self, *, valid=True):
        self.valid = valid

    def save_to_path(self, path):
        checkpoint = Path(path)
        checkpoint.mkdir(parents=True)
        if self.valid:
            (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "algorithm_state.pkl").write_bytes(b"fake")
        return str(checkpoint)


class _FakeDeterministicDistribution:
    def __init__(self, logits):
        self.logits = logits

    @classmethod
    def from_logits(cls, logits):
        return cls(logits)

    def to_deterministic(self):
        return self

    def sample(self):
        # Deliberately choose the second distribution input. This catches the
        # old evaluator's incorrect "first half of logits == action" shortcut.
        return self.logits[:, 1:2]


class _DistributionModule(_FakeModule):
    def forward_inference(self, batch):
        import torch

        assert tuple(batch["obs"].shape) == (1, 2)
        return {
            "action_dist_inputs": torch.tensor([[0.1, 0.75]], dtype=torch.float32)
        }

    def get_inference_action_dist_cls(self):
        return _FakeDeterministicDistribution


class IPPOInfrastructureTest(unittest.TestCase):
    def test_protocol_eval_schedule_uses_threshold_crossing_not_modulo(self):
        crossed, next_step = common.crossed_eval_targets(48_000, 50_000, 50_000)
        self.assertEqual(crossed, [])
        self.assertEqual(next_step, 50_000)

        crossed, next_step = common.crossed_eval_targets(52_000, next_step, 50_000)
        self.assertEqual(crossed, [50_000])
        self.assertEqual(next_step, 100_000)

        crossed, next_step = common.crossed_eval_targets(96_000, next_step, 50_000)
        self.assertEqual(crossed, [])
        crossed, next_step = common.crossed_eval_targets(104_000, next_step, 50_000)
        self.assertEqual(crossed, [100_000])
        self.assertEqual(next_step, 150_000)

    def test_canonical_budget_has_exactly_one_hundred_eval_targets(self):
        targets = []
        next_step = 50_000
        for current_step in range(4_000, 5_000_001, 4_000):
            crossed, next_step = common.crossed_eval_targets(
                current_step, next_step, 50_000
            )
            self.assertLessEqual(len(crossed), 1)
            targets.extend(crossed)
        self.assertEqual(targets, list(range(50_000, 5_000_001, 50_000)))
        self.assertEqual(len(targets), 100)

    def test_resumed_periodic_schedule_skips_completed_thresholds(self):
        self.assertEqual(common.next_periodic_target(0, 1_000), 1_000)
        self.assertEqual(common.next_periodic_target(1_000, 1_000), 2_000)
        self.assertEqual(common.next_periodic_target(4_008, 1_000), 5_000)
        self.assertEqual(common.next_periodic_target(4_008, 5_000), 5_000)
        with self.assertRaises(ValueError):
            common.next_periodic_target(-1, 1_000)
        with self.assertRaises(ValueError):
            common.next_periodic_target(0, 0)

    def test_extract_env_steps_rejects_ambiguous_agent_counter(self):
        self.assertEqual(
            common.extract_env_steps({"num_env_steps_sampled_lifetime": 123}), 123
        )
        with self.assertRaisesRegex(RuntimeError, "no num_env_steps_sampled_lifetime"):
            common.extract_env_steps({"num_agent_steps_sampled_lifetime": 246})

    def test_rllib_global_info_translation_and_conflict(self):
        original = {"left": {}, "right": {}, "_global_": TERMINAL_METRICS}
        translated = common.normalize_rllib_infos(original)
        self.assertNotIn("_global_", translated)
        self.assertEqual(translated["__common__"], TERMINAL_METRICS)
        self.assertIn("_global_", original)

        with self.assertRaisesRegex(RuntimeError, "both"):
            common.normalize_rllib_infos(
                {"_global_": TERMINAL_METRICS, "__common__": TERMINAL_METRICS}
            )

    def test_eval_counts_shared_reward_once_and_collects_terminal_f1(self):
        env = _TwoStepSharedEnv()
        result = common.evaluate_marl(
            object(),
            "fake-env",
            eval_seed=10_000,
            num_episodes=2,
            env_factory=lambda env_id, **kwargs: env,
            action_computer=_zero_action,
        )
        self.assertTrue(env.closed)
        self.assertEqual(env.reset_seeds, [10_000, 20_000])
        self.assertEqual(
            [episode["team_return"] for episode in result["episodes"]],
            [2.0, 2.0],
        )
        self.assertEqual(
            result["episodes"][0]["per_agent_returns"],
            {"left": 2.0, "right": 2.0},
        )
        self.assertEqual(result["summary"]["team_return_mean"], 2.0)
        self.assertEqual(
            result["summary"]["episode_task/musical_f1_mean"], 0.5
        )
        self.assertEqual(
            result["summary"][
                "episode_coordination/common_area_success_rate_mean"
            ],
            0.75,
        )

    def test_eval_forwards_video_configuration_only_when_recording(self):
        env = _TwoStepSharedEnv()
        received = {}

        def _factory(env_id, **kwargs):
            self.assertEqual(env_id, "fake-env")
            received.update(kwargs)
            return env

        common.evaluate_marl(
            object(),
            "fake-env",
            eval_seed=0,
            num_episodes=1,
            record_dir="/tmp/eval-video",
            record_resolution=(360, 640),
            camera_id="piano/topdown",
            inter_agent_collision_penalty_coef=0.25,
            env_factory=_factory,
            action_computer=_zero_action,
        )
        self.assertEqual(received["record_dir"], "/tmp/eval-video")
        self.assertEqual(received["record_every"], 1)
        self.assertEqual(received["record_resolution"], (360, 640))
        self.assertEqual(received["camera_id"], "piano/topdown")
        self.assertEqual(received["inter_agent_collision_penalty_coef"], 0.25)

    def test_eval_rejects_invalid_inter_agent_collision_penalty(self):
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                common.evaluate_marl(
                    object(),
                    "fake-env",
                    eval_seed=0,
                    num_episodes=1,
                    inter_agent_collision_penalty_coef=invalid,
                    env_factory=lambda env_id, **kwargs: _TwoStepSharedEnv(),
                    action_computer=_zero_action,
                )

    def test_eval_fails_when_terminal_f1_is_missing_and_closes_env(self):
        env = _TwoStepSharedEnv(include_metrics=False)
        with self.assertRaisesRegex(RuntimeError, "musical metrics"):
            common.evaluate_marl(
                object(),
                "fake-env",
                eval_seed=0,
                num_episodes=1,
                env_factory=lambda env_id, **kwargs: env,
                action_computer=_zero_action,
            )
        self.assertTrue(env.closed)

    def test_terminal_coordination_rates_are_range_checked(self):
        invalid = dict(TERMINAL_METRICS)
        invalid["episode_coordination/common_area_duplicate_press_rate"] = 1.1
        with self.assertRaisesRegex(RuntimeError, "terminal rate metrics"):
            common.extract_terminal_metrics({"_global_": invalid})

    def test_eval_rejects_non_shared_rewards(self):
        env = _TwoStepSharedEnv(unequal_rewards=True)
        with self.assertRaisesRegex(RuntimeError, "different rewards"):
            common.evaluate_marl(
                object(),
                "fake-env",
                eval_seed=0,
                num_episodes=1,
                env_factory=lambda env_id, **kwargs: env,
                action_computer=_zero_action,
            )

    def test_inference_exception_is_contextual_and_never_zero_action(self):
        # A cleanup failure must not replace the actual inference exception.
        env = _TwoStepSharedEnv(close_raises=True)

        def _broken_action(agent, observation, action_space):
            del agent, observation, action_space
            raise ValueError("broken checkpoint")

        with self.assertRaisesRegex(
            RuntimeError, "episode=0, step=0, agent='left'"
        ) as caught:
            common.evaluate_marl(
                object(),
                "fake-env",
                eval_seed=0,
                num_episodes=1,
                env_factory=lambda env_id, **kwargs: env,
                action_computer=_broken_action,
            )
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(env.step_count, 0)
        self.assertTrue(env.closed)

    def test_action_validation_is_fail_fast(self):
        cases = [
            ([0.0, 0.0], "action shape"),
            ([float("nan")], "non-finite action"),
        ]
        for action, error in cases:
            with self.subTest(action=action):
                computer = common.DeterministicActionComputer(
                    _FakeAlgorithm(action), ["left"]
                )
                with self.assertRaisesRegex(RuntimeError, error):
                    computer("left", np.zeros(2, dtype=np.float32), _Box())

    def test_rlmodule_distribution_uses_deterministic_distribution_api(self):
        algo = _FakeAlgorithm([0.0])
        algo.module = _DistributionModule([0.0])
        computer = common.DeterministicActionComputer(algo, ["left"])
        action = computer("left", np.zeros(2, dtype=np.float32), _Box())
        np.testing.assert_allclose(action, np.array([0.75], dtype=np.float32))

    def test_checkpoint_reference_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            self.assertEqual(
                common.checkpoint_reference(checkpoint, root), "checkpoint"
            )

    def test_checkpoint_save_requires_a_real_rllib_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = common.save_algorithm_checkpoint(
                _FakeCheckpointAlgorithm(), root / "valid"
            )
            self.assertEqual(Path(checkpoint), (root / "valid").resolve())
            with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
                common.save_algorithm_checkpoint(
                    _FakeCheckpointAlgorithm(valid=False), root / "invalid"
                )

if __name__ == "__main__":
    unittest.main()
