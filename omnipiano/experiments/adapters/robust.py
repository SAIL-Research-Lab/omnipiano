"""Adapter for the existing TorchRL Robust RL entrypoint."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from omnipiano.experiments.adapters.common import (
    AdapterError,
    append_option,
    backend_device,
    reject_compute_fields,
    reject_unknown,
    require_default_preset,
    require_evaluation,
    require_registered_task,
    require_single_agent_environment,
)
from omnipiano.experiments.models import ResolvedExperiment, RunPlan


BACKEND = "torchrl"
ENTRYPOINT = "examples/run_torchrl_robust.py"
NETWORK_PARAMETERS = {"net_arch"}
PPO_PARAMETERS = {
    "n_steps", "n_epochs", "gae_lambda", "clip_epsilon",
    "entropy_coeff", "critic_coeff",
}
SAC_PARAMETERS = {
    "batch_size", "buffer_size", "learning_starts", "tau", "utd",
    "learning_rate",
}
ALGORITHM_PARAMETERS = {
    "ppo": NETWORK_PARAMETERS | PPO_PARAMETERS,
    "sac": NETWORK_PARAMETERS | SAC_PARAMETERS,
    "eppo": NETWORK_PARAMETERS | PPO_PARAMETERS | {
        "eppo_mode", "kappa", "evidential_reg",
    },
    "a2p_sac": NETWORK_PARAMETERS | SAC_PARAMETERS | {
        "a2p_epsilon_mode", "a2p_epsilon", "adversary_learning_rate",
        "adversary_update_freq", "a2p_warmup_steps",
    },
    "ompo": NETWORK_PARAMETERS | SAC_PARAMETERS | {
        "ompo_local_buffer_size", "ompo_updates_per_step",
        "ompo_discriminator_observation", "ompo_exponent",
        "ompo_correction_coeff", "ompo_reward_max",
    },
    "scpo": NETWORK_PARAMETERS | SAC_PARAMETERS | {
        "scpo_state_noise", "scpo_gradient_observation",
    },
}


def bind_run(resolved: ResolvedExperiment, run: RunPlan) -> RunPlan:
    document = resolved.to_dict()
    require_default_preset(document, track="robust")
    reject_unknown(
        document["design"]["config"],
        {"evaluation_protocol"},
        context="robust design.config",
    )
    protocol = document["design"]["config"].get("evaluation_protocol", "matched")
    if protocol != "matched":
        raise AdapterError(
            "the current Robust RL entrypoint supports matched evaluation only"
        )
    env_id = require_registered_task(document, track="robust")
    require_single_agent_environment(env_id, track="robust")
    evaluation = require_evaluation(document, track="robust")
    algorithm = document["algorithm"]["name"]
    parameters: Mapping[str, Any] = document["algorithm"]["parameters"]
    allowed = ALGORITHM_PARAMETERS[algorithm]
    reject_unknown(parameters, allowed, context="robust algorithm.parameters")

    compute = document["run"]["compute"]
    reject_compute_fields(compute, {"cpus", "gpus"}, track="robust")
    if compute["num_workers"] != 1:
        raise AdapterError(
            "Robust RL uses run.compute.num_envs; num_workers must remain 1"
        )
    if document["run"]["tracking"]["provider"] != "none":
        raise AdapterError("the Robust RL backend exposes CSV logging only")
    if document["run"]["video"]["enabled"]:
        raise AdapterError(
            "the Robust RL backend records final evaluation video only; "
            "periodic unified video settings are unsupported"
        )
    command = [
        "python", ENTRYPOINT,
        "--algo", algorithm,
        "--env", env_id,
        "--experiment-name", run.run_id,
        "--logs", run.output_dir,
        "--seed", str(run.seed),
        "--total-steps", str(document["run"]["total_steps"]),
        "--gamma", str(document["run"]["gamma"]),
        "--eval-freq", str(evaluation["frequency"]),
        "--num-eval-eps", str(evaluation["episodes"]),
        "--n-envs", str(compute["num_envs"]),
        "--device", backend_device(compute["device"], auto="cpu"),
    ]
    for name in sorted(parameters):
        append_option(command, name, parameters[name])
    return replace(
        run,
        launch_status="bound",
        backend=BACKEND,
        command=tuple(command),
    )
