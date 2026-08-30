"""Fast regression tests for the canonical MARL command structure."""

from __future__ import annotations

import inspect

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
    assert args.wandb_entity == "omnipiano"
    assert args.wandb_project == "marl"
    assert inspect.signature(WandbRun).parameters["project"].default == "marl"


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
