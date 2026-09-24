"""Fast regression tests for the canonical MARL command structure."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import omnipiano
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent import train
from omnipiano.multiagent.training.tracking import WandbRun
from omnipiano.multiagent.algos import get_algo


def test_canonical_defaults_match_checked_in_project_convention() -> None:
    proto = BenchmarkProtocolConfig()
    args = train.build_arg_parser().parse_args([])
    # The project intentionally trains for 10M; the frozen benchmark protocol
    # remains 5M and run_config records this as a non-protocol budget.
    assert args.total_steps == 10_000_000
    assert proto.total_env_steps == 5_000_000
    assert args.gamma == proto.gamma == 0.8
    assert args.eval_freq == proto.eval_freq_env_steps == 50_000
    assert args.inter_agent_collision_penalty_coef == 0.1
    assert args.seed == proto.seed == 0
    assert args.checkpoint_freq == 500_000
    assert args.video_enabled
    assert args.video_freq == 500_000
    assert args.video_record_final
    assert args.video_camera_id == "piano/topdown"
    assert (args.video_height, args.video_width) == (480, 640)
    assert args.video_wandb_upload
    assert args.log_every_iters == 5
    assert not args.dry_run
    assert args.train_batch_size == args.minibatch_size == 4_000
    assert args.num_epochs == 5
    assert args.lr == 3e-4
    assert args.critic_lr == 3e-4
    assert args.adam_epsilon == 1e-5
    assert args.gae_lambda == 0.95
    assert args.clip_param == args.vf_clip_param == 0.2
    assert args.entropy_coeff == 0.0
    assert not args.use_kl_loss
    assert args.grad_clip == 10.0
    assert args.grad_clip_by == "global_norm"
    assert args.hidden_orthogonal_gain == 2 ** 0.5
    assert args.policy_output_gain == 0.01
    assert args.value_output_gain == 1.0
    assert args.initial_log_std == 0.0
    assert (args.log_std_min, args.log_std_max) == (-5.0, 2.0)
    assert args.input_layer_norm
    assert args.value_norm
    assert args.value_norm_variance_floor == 0.01
    assert args.max_sampling_stalls == 3
    assert args.num_workers == 9
    assert args.num_gpus_per_learner == 1.0
    assert args.ray_num_cpus == 11
    assert Path(args.config).resolve() == train.DEFAULT_TRAIN_CONFIG_PATH.resolve()
    assert args._training_config_snapshot["schema_version"] == 1
    assert args.wandb_entity == "omnipiano"
    assert args.wandb_project == "multiagent"
    assert inspect.signature(WandbRun).parameters["project"].default == "multiagent"


def test_dry_run_exits_before_training_side_effects(monkeypatch, capsys) -> None:
    space = SimpleNamespace(shape=(3,), dtype=np.dtype(np.float32))
    monkeypatch.setattr(
        train,
        "probe_agent_spaces",
        lambda *args, **kwargs: (
            ["secondo", "primo"],
            {"secondo": space, "primo": space},
            {"secondo": space, "primo": space},
            {
                "secondo": {"own": (0, 3), "global_state": (0, 0)},
                "primo": {"own": (0, 3), "global_state": (0, 0)},
            },
        ),
    )

    assert train.main(["--dry-run", "--num-gpus-per-learner", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["gpu_or_cluster_started"] is False
    assert report["resolved_task"]["legacy_env_id"] == report["env_id"]


def test_registered_env_dry_run_applies_json_observation_noise(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    config = json.loads(train.DEFAULT_TRAIN_CONFIG_PATH.read_text())
    config["robust"] = {
        "noise_dist": "gaussian",
        "obs_noise_std": 0.05,
    }
    path = tmp_path / "registered-obs-noise.json"
    path.write_text(json.dumps(config))
    space = SimpleNamespace(shape=(3,), dtype=np.dtype(np.float32))
    monkeypatch.setattr(
        train,
        "probe_agent_spaces",
        lambda *args, **kwargs: (
            ["secondo", "primo"],
            {"secondo": space, "primo": space},
            {"secondo": space, "primo": space},
            {
                "secondo": {"own": (0, 3), "global_state": (0, 0)},
                "primo": {"own": (0, 3), "global_state": (0, 0)},
            },
        ),
    )

    assert train.main([str(path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    robust = report["resolved_task"]["robust_config"]
    assert robust["noise_dist"] == "gaussian"
    assert robust["obs_noise_std"] == 0.05
    assert robust["action_noise_std"] == 0.0
    assert robust["reward_noise_std"] == 0.0


def test_json_algorithm_defaults_then_cli_overrides(tmp_path: Path) -> None:
    config = json.loads(train.DEFAULT_TRAIN_CONFIG_PATH.read_text())
    config["ppo"]["entropy_coeff"] = 0.0001
    config["reward"]["inter_agent_collision_penalty_coef"] = 0.25
    config["algorithm_overrides"]["mappo"] = {
        "ppo": {"num_epochs": 7}
    }
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(config))

    args = train._parse_args([
        "--config", str(path), "--algo", "mappo", "--num-epochs", "9",
        "--inter-agent-collision-penalty-coef", "0.4",
    ])
    assert args.algo == "mappo"
    assert args.entropy_coeff == 0.0001  # JSON shared default.
    assert args.num_epochs == 9  # CLI wins over algorithm JSON override (7).
    assert args.inter_agent_collision_penalty_coef == 0.4
    assert args._training_config_path == str(path.resolve())


def test_legacy_json_without_reward_section_preserves_original_reward(
    tmp_path: Path,
) -> None:
    config = json.loads(train.DEFAULT_TRAIN_CONFIG_PATH.read_text())
    del config["reward"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(config))

    args = train._parse_args(["--config", str(path)])
    assert args.inter_agent_collision_penalty_coef == 0.0


def test_inter_agent_collision_penalty_validation() -> None:
    args = train._parse_args([
        "--inter-agent-collision-penalty-coef", "-0.1"
    ])
    try:
        train._resolve_args(args)
    except ValueError as exc:
        assert "finite and non-negative" in str(exc)
    else:
        raise AssertionError("negative collision penalty coefficient was accepted")


def test_rllib_debug_anchor_has_explicit_stock_value_loss_override() -> None:
    args = train._parse_args(["--algo", "ippo-rllib-module"])
    spec = train._resolve_args(args)
    assert spec.rl_module == "rllib_default"
    assert args.vf_clip_param == 1000.0


def test_smoke_test_values_also_come_from_json() -> None:
    args = train._parse_args(["--smoke-test"])
    train._resolve_args(args)
    assert args.total_steps == args.eval_freq == 5_000
    assert args.train_batch_size == args.minibatch_size == 512
    assert args.num_epochs == 2
    assert args.num_workers == 0
    assert args.num_gpus_per_learner == 0.0
    assert args.ray_num_cpus is None
    assert args.checkpoint_freq == 0
    assert not args.video_enabled

    overridden = train._parse_args(["--smoke-test", "--num-epochs", "3"])
    train._resolve_args(overridden)
    assert overridden.num_epochs == 3  # Explicit CLI remains highest priority.


def test_ippo_and_mappo_are_a_single_factor_critic_ablation() -> None:
    ippo = get_algo("ippo")
    mappo = get_algo("mappo")
    assert ippo.critic_input == "own"
    assert not ippo.needs_global_state
    assert mappo.critic_input == "global"
    assert mappo.needs_global_state
    assert ippo.rl_module == mappo.rl_module == "ctde"
    assert ippo.training_overrides == mappo.training_overrides == {}


def test_top_level_multiagent_entrypoints_are_real() -> None:
    assert omnipiano.make_parallel is not None
    assert omnipiano.register_parallel is not None
    assert omnipiano.list_parallel_envs is not None
