"""Fast regression tests for the canonical MARL command structure."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import omnipiano
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent import evaluate, train
from omnipiano.multiagent._wandb import WandbRun
from omnipiano.multiagent.algos import get_algo


def test_canonical_defaults_match_protocol_and_project_convention() -> None:
    proto = BenchmarkProtocolConfig()
    args = train.build_arg_parser().parse_args([])
    assert args.total_steps == proto.total_env_steps == 5_000_000
    assert args.gamma == proto.gamma == 0.8
    assert args.eval_freq == proto.eval_freq_env_steps == 50_000
    assert args.seed == proto.seed == 0
    assert args.checkpoint_freq == 500_000
    assert args.log_every_iters == 5
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


def test_json_algorithm_defaults_then_cli_overrides(tmp_path: Path) -> None:
    config = json.loads(train.DEFAULT_TRAIN_CONFIG_PATH.read_text())
    config["ppo"]["entropy_coeff"] = 0.0001
    config["algorithm_overrides"]["mappo"] = {
        "ppo": {"num_epochs": 7}
    }
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(config))

    args = train._parse_args([
        "--config", str(path), "--algo", "mappo", "--num-epochs", "9",
    ])
    assert args.algo == "mappo"
    assert args.entropy_coeff == 0.0001  # JSON shared default.
    assert args.num_epochs == 9  # CLI wins over algorithm JSON override (7).
    assert args._training_config_path == str(path.resolve())


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

    overridden = train._parse_args(["--smoke-test", "--num-epochs", "3"])
    train._resolve_args(overridden)
    assert overridden.num_epochs == 3  # Explicit CLI remains highest priority.


def test_canonical_evaluator_uses_protocol_episode_default() -> None:
    proto = BenchmarkProtocolConfig()
    args = evaluate.build_arg_parser().parse_args(["--checkpoint", "somewhere"])
    assert args.num_eval_eps == proto.num_eval_eps


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
