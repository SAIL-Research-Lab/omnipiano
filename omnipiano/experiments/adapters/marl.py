"""Adapter for the existing MARL JSON compiler and training entrypoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePath
import re
from typing import Any, Mapping

from omnipiano.experiments.adapters.common import (
    AdapterError,
    append_option,
    reject_unknown,
    require_default_preset,
)
from omnipiano.experiments.models import ResolvedExperiment, RunPlan


BACKEND = "omnipiano_marl"
ENTRYPOINT = "omnipiano.multiagent.train"
DEFAULT_CONFIG = "omnipiano/multiagent/configs/marl_train_config_default.json"
PARAMETER_FLAGS = {
    "learning_rate": "lr",
    "critic_learning_rate": "critic_lr",
    "train_batch_size": "train_batch_size",
    "minibatch_size": "minibatch_size",
    "num_epochs": "num_epochs",
    "adam_epsilon": "adam_epsilon",
    "gae_lambda": "gae_lambda",
    "clip_param": "clip_param",
    "value_clip_param": "vf_clip_param",
    "value_loss_coefficient": "vf_loss_coeff",
    "entropy_coefficient": "entropy_coeff",
    "use_kl_loss": "use_kl_loss",
    "gradient_clip": "grad_clip",
    "gradient_clip_mode": "grad_clip_by",
    "hidden_sizes": "hidden_sizes",
    "activation": "activation",
    "input_layer_norm": "input_layer_norm",
    "value_norm": "value_norm",
}
PPO_PARAMETER_ALGORITHMS = {
    "ippo", "ippo-rllib-module", "mappo", "mappo-own-critic",
    "happo", "a2po", "ppo-monolithic",
}


def _custom_task_config(document: Mapping[str, Any], run: RunPlan) -> dict[str, Any]:
    task = document["task"]
    music = task["music"]
    if music["source"] != "repertoire":
        raise AdapterError("the current MARL compiler does not support custom MIDI paths")
    if any(
        music[key] != expected
        for key, expected in (
            ("trim_silence", True), ("stretch_factor", 1.0),
            ("shift_factor", 0), ("fingering_mode", "optimal_transport"),
        )
    ):
        raise AdapterError(
            "the current MARL compiler cannot apply music preprocessing overrides"
        )
    embodiment = task["embodiment"]
    num_hands = embodiment["num_hands"]
    if num_hands not in {3, 4, 5}:
        raise AdapterError("the current MARL custom-task compiler supports 3, 4, or 5 hands")
    if embodiment["mobility"] not in {"unrestricted", "custom"}:
        raise AdapterError("the current MARL compiler does not support static_partition mobility")

    from omnipiano.multiagent.compile.presets import LAYOUTS

    morphology, hand_names, buckets = LAYOUTS[num_hands]
    if embodiment["layout_preset"] != morphology:
        raise AdapterError(
            f"the {num_hands}-hand MARL backend requires "
            f"layout_preset={morphology!r}"
        )
    hands = sorted(embodiment["hands"], key=lambda hand: hand["id"])
    if [hand["id"] for hand in hands] != list(range(num_hands)):
        raise AdapterError("MARL custom-task hand IDs must be contiguous from zero")
    for hand, expected_name, bucket in zip(hands, hand_names, buckets):
        expected_range = [bucket[0] + 1, bucket[1] + 1]
        expected_side = "left" if expected_name.startswith("lh") else "right"
        if (
            hand["name"] != expected_name
            or hand["side"] != expected_side
            or not hand["mobile"]
            or hand["initial_key_range"] != expected_range
        ):
            raise AdapterError(
                f"hand {hand['id']} does not match the fixed "
                f"{morphology} backend layout"
            )

    config = dict(document["design"]["config"])
    allowed = {"num_agents", "assignment", "sustain_owner", "agents", "coordination"}
    reject_unknown(config, allowed, context="MARL design.config")
    piece = music["piece"]
    if not isinstance(piece, str) or re.fullmatch(r"[A-Za-z0-9]+", piece) is None:
        raise AdapterError("MARL repertoire piece must be an alphanumeric token")
    backend_task: dict[str, Any] = {
        "name": re.sub(r"[^A-Za-z0-9_-]", "-", document["experiment"]["name"]),
        "song": piece,
        "num_hands": num_hands,
    }
    for key in ("num_agents", "assignment", "sustain_owner", "agents"):
        if key in config:
            backend_task[key] = config[key]
    if backend_task.get("assignment") == "territorial":
        backend_task["assignment"] = "default"

    result: dict[str, Any] = {
        "schema_version": 2,
        "description": document["experiment"]["description"],
        "task": backend_task,
        "experiment": {
            "algo": document["algorithm"]["name"],
            "env_id": None,
            "seed": run.seed,
            "run_dir": run.output_dir,
        },
    }
    coordination = config.get("coordination", {})
    if coordination:
        reject_unknown(
            coordination,
            {"inter_agent_collision_penalty_coefficient"},
            context="MARL design.config.coordination",
        )
        if "inter_agent_collision_penalty_coefficient" in coordination:
            result["reward"] = {
                "inter_agent_collision_penalty_coef": coordination[
                    "inter_agent_collision_penalty_coefficient"
                ]
            }
    return result


def bind_run(resolved: ResolvedExperiment, run: RunPlan) -> RunPlan:
    document = resolved.to_dict()
    require_default_preset(document, track="marl")
    task = document["task"]
    artifacts: dict[str, Any] = {}
    if task["source"] == "registered":
        if task.get("overrides"):
            raise AdapterError(
                "the MARL registered-task adapter cannot apply task.overrides"
            )
        from omnipiano.multiagent import list_parallel_envs

        if task["environment_id"] not in set(list_parallel_envs()):
            raise AdapterError(
                f"{task['environment_id']!r} is not a registered MARL environment"
            )
        if document["algorithm"]["name"] == "ppo-monolithic":
            raise AdapterError(
                "ppo-monolithic requires a custom MARL task with num_agents=1"
            )
        design = document["design"]["config"]
        reject_unknown(design, {"assignment"}, context="MARL design.config")
        assignment = design.get("assignment")
        if assignment is not None and assignment.lower() not in task[
            "environment_id"
        ].lower():
            raise AdapterError(
                "MARL design.config.assignment does not match environment_id"
            )
        config_path = DEFAULT_CONFIG
    else:
        required_agents = 1 if document["algorithm"]["name"] == "ppo-monolithic" else None
        requested_agents = document["design"]["config"].get("num_agents")
        if required_agents is not None and requested_agents != required_agents:
            raise AdapterError("ppo-monolithic requires design.config.num_agents=1")
        config_path = str(PurePath(run.output_dir) / "backend_config.json")
        artifacts[config_path] = _custom_task_config(document, run)

    evaluation = document["run"]["evaluation"]
    if not evaluation["enabled"]:
        raise AdapterError("the MARL backend cannot disable evaluation")
    compute = document["run"]["compute"]
    if compute["num_envs"] != 1:
        raise AdapterError("the MARL backend currently requires run.compute.num_envs=1")
    parameters: Mapping[str, Any] = document["algorithm"]["parameters"]
    allowed_parameters = (
        PARAMETER_FLAGS
        if document["algorithm"]["name"] in PPO_PARAMETER_ALGORITHMS
        else {}
    )
    reject_unknown(
        parameters, allowed_parameters, context="MARL algorithm.parameters"
    )

    command = [
        "python", "-m", ENTRYPOINT, config_path,
        "--algo", document["algorithm"]["name"],
        "--seed", str(run.seed),
        "--run-dir", run.output_dir,
        "--total-steps", str(document["run"]["total_steps"]),
        "--gamma", str(document["run"]["gamma"]),
        "--eval-freq", str(evaluation["frequency"]),
        "--num-eval-eps", str(evaluation["episodes"]),
        "--eval-seed-offset", str(evaluation["seed_offset"]),
        "--num-workers", str(compute["num_workers"]),
    ]
    if task["source"] == "registered":
        command.extend(("--env-id", task["environment_id"]))
    if "cpus" in compute:
        command.extend(("--ray-num-cpus", str(int(compute["cpus"]))))
    if "gpus" in compute:
        command.extend(("--num-gpus-per-learner", str(compute["gpus"])))

    video = document["run"]["video"]
    command.append("--eval-video" if video["enabled"] else "--no-eval-video")
    if video["enabled"]:
        command.extend((
            "--video-freq", str(video["frequency"]),
            "--video-camera-id", video["camera"],
            "--video-width", str(video["width"]),
            "--video-height", str(video["height"]),
        ))

    tracking = document["run"]["tracking"]
    provider = tracking["provider"]
    if provider == "tensorboard":
        raise AdapterError("the MARL backend does not expose TensorBoard tracking")
    wandb_mode = tracking["mode"] if provider == "wandb" else "disabled"
    command.extend(("--wandb-mode", wandb_mode))
    for key, flag in (
        ("project", "wandb-project"), ("group", "wandb-group"),
        ("name", "wandb-name"),
    ):
        if tracking[key] is not None:
            command.extend(("--" + flag, str(tracking[key])))
    if tracking["tags"]:
        command.extend(("--wandb-tags", ",".join(tracking["tags"])))

    for name, value in parameters.items():
        append_option(command, PARAMETER_FLAGS[name], value)
    environment = {}
    if compute["device"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif compute["device"] not in {"auto", "cuda"}:
        raise AdapterError(
            f"unsupported MARL compute device {compute['device']!r}; use auto, cpu, or cuda"
        )
    return replace(
        run,
        launch_status="bound",
        backend=BACKEND,
        command=tuple(command),
        environment=environment,
        artifacts=artifacts,
    )
