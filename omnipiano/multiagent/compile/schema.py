"""Data contracts for compiling an OmniPiano MARL experiment.

These types deliberately contain no Ray, MuJoCo, PettingZoo, or W&B objects.
They describe the user's versioned JSON request and the deterministic result of
selecting one algorithm, task layout, ownership, and range configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


TRAIN_CONFIG_SCHEMA_VERSION = 1
TASK_CONFIG_SCHEMA_VERSION = 2

# JSON keys are mapped to the argparse destinations consumed by the existing
# trainer. Keeping this translation at the compile boundary prevents train.py
# from knowing the on-disk JSON layout.
CONFIG_FIELDS: Dict[str, Dict[str, str]] = {
    "experiment": {
        "algo": "algo",
        "env_id": "env_id",
        "seed": "seed",
        "run_dir": "run_dir",
    },
    "protocol": {
        "total_steps": "total_steps",
        "gamma": "gamma",
        "eval_freq": "eval_freq",
        "num_eval_eps": "num_eval_eps",
        "eval_seed_offset": "eval_seed_offset",
    },
    "reward": {
        "inter_agent_collision_penalty_coef": (
            "inter_agent_collision_penalty_coef"
        ),
    },
    "ppo": {
        "train_batch_size": "train_batch_size",
        "minibatch_size": "minibatch_size",
        "num_epochs": "num_epochs",
        "lr": "lr",
        "critic_lr": "critic_lr",
        "adam_epsilon": "adam_epsilon",
        "gae_lambda": "gae_lambda",
        "clip_param": "clip_param",
        "vf_clip_param": "vf_clip_param",
        "vf_loss_coeff": "vf_loss_coeff",
        "entropy_coeff": "entropy_coeff",
        "use_kl_loss": "use_kl_loss",
        "grad_clip": "grad_clip",
        "grad_clip_by": "grad_clip_by",
    },
    "network": {
        "hidden_sizes": "hidden_sizes",
        "activation": "activation",
        "hidden_orthogonal_gain": "hidden_orthogonal_gain",
        "policy_output_gain": "policy_output_gain",
        "value_output_gain": "value_output_gain",
        "initial_log_std": "initial_log_std",
        "log_std_min": "log_std_min",
        "log_std_max": "log_std_max",
        "input_layer_norm": "input_layer_norm",
        "value_norm": "value_norm",
        "value_norm_beta": "value_norm_beta",
        "value_norm_epsilon": "value_norm_epsilon",
        "value_norm_variance_floor": "value_norm_variance_floor",
    },
    "compute": {
        "num_workers": "num_workers",
        "num_cpus_per_env_runner": "num_cpus_per_env_runner",
        "num_learners": "num_learners",
        "num_gpus_per_learner": "num_gpus_per_learner",
        "ray_num_cpus": "ray_num_cpus",
        "sample_timeout_s": "sample_timeout_s",
        "max_sampling_stalls": "max_sampling_stalls",
        "ray_log_to_driver": "ray_log_to_driver",
        "checkpoint_freq": "checkpoint_freq",
        "log_every_iters": "log_every_iters",
        "smoke_test": "smoke_test",
    },
    "video": {
        "enabled": "video_enabled",
        "freq": "video_freq",
        "record_final": "video_record_final",
        "camera_id": "video_camera_id",
        "height": "video_height",
        "width": "video_width",
        "wandb_upload": "video_wandb_upload",
    },
    "wandb": {
        "mode": "wandb_mode",
        "entity": "wandb_entity",
        "project": "wandb_project",
        "group": "wandb_group",
        "name": "wandb_name",
        "tags": "wandb_tags",
        "notes": "wandb_notes",
        "upload_artifacts": "wandb_upload_artifacts",
    },
}

# Robustness is part of the environment identity rather than a trainer CLI
# default.  MARL currently implements the observation channel only, so the
# public JSON contract deliberately does not accept action/reward fields that
# the runtime would otherwise have to ignore.
ROBUST_OBSERVATION_FIELDS = frozenset({
    "noise_dist",
    "obs_noise_dist",
    "obs_noise_std",
    "obs_noise_uniform_low",
    "obs_noise_uniform_high",
    "obs_noise_shift",
})


@dataclass(frozen=True)
class ExperimentRequest:
    """Validated representation of one versioned user JSON document."""

    schema_version: int
    description: str
    sections: Mapping[str, Mapping[str, Any]]
    smoke_test_overrides: Mapping[str, Any]
    algorithm_overrides: Mapping[str, Any]
    _snapshot: Mapping[str, Any]

    def snapshot(self) -> Dict[str, Any]:
        """Return an independent copy suitable for an experiment artifact."""
        return deepcopy(dict(self._snapshot))


@dataclass(frozen=True)
class ResolvedHand:
    """Hand IDs are zero-based, in the preset's spatial left-to-right order."""

    id: int
    name: str
    bucket_key_range: Tuple[int, int]  # Internal inclusive 0..87 indices.


@dataclass(frozen=True)
class ResolvedAgent:
    name: str
    hand_ids: Tuple[int, ...]
    action_key_range: Tuple[int, int]  # Wrist-slider limits, NOT fingertip limits.
    observation_key_range: Optional[Tuple[int, int]]
    visible_teammate_hands: Tuple[str, ...]
    is_sustain_owner: bool = False


@dataclass(frozen=True)
class ResolvedTask:
    """Serializable task contract shared by training workers and evaluation.

    Pure compilation resolves ownership and ranges. ``prepare_task`` fills the
    physics/config snapshots from the existing layout registry before launch;
    workers then consume those snapshots rather than a driver-only registration.
    ``observation_key_range=None`` is reserved for historical reach-based obs.
    """

    name: str
    song: str
    base_env_name: str
    layout: str
    assignment_mode: str
    hands: Tuple[ResolvedHand, ...]
    agents: Tuple[ResolvedAgent, ...]
    legacy_env_id: Optional[str] = None
    hand_specs: Tuple[Mapping[str, Any], ...] = ()
    env_config: Optional[Mapping[str, Any]] = None
    task_config: Optional[Mapping[str, Any]] = None
    robust_config: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolvedTask":
        """Restore our own JSON snapshot (not a second user-input format)."""
        raw = deepcopy(dict(data))
        raw["hands"] = tuple(
            ResolvedHand(int(h["id"]), h["name"], tuple(h["bucket_key_range"]))
            for h in raw["hands"]
        )
        raw["agents"] = tuple(
            ResolvedAgent(
                name=a["name"], hand_ids=tuple(a["hand_ids"]),
                action_key_range=tuple(a["action_key_range"]),
                observation_key_range=(tuple(a["observation_key_range"])
                                       if a["observation_key_range"] is not None
                                       else None),
                visible_teammate_hands=tuple(a["visible_teammate_hands"]),
                is_sustain_owner=a["is_sustain_owner"],
            ) for a in raw["agents"]
        )
        raw["hand_specs"] = tuple(raw.get("hand_specs", ()))
        return cls(**raw)


@dataclass(frozen=True)
class ResolvedExperiment:
    """JSON defaults after selecting and applying one algorithm override.

    Explicit command-line values and smoke-test overrides are intentionally not
    applied here: argparse still owns that last precedence layer in Phase 2.
    ``values`` uses the trainer's established argparse destination names, so the
    training path is unchanged while its file-format knowledge moves out.
    """

    request: ExperimentRequest
    config_path: Path
    algorithm: str
    values: Mapping[str, Any]
    smoke_test_overrides: Mapping[str, Any]
    native_options: Mapping[str, Any]
    task: Optional[ResolvedTask] = None
    robust_config: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a fully JSON-serializable description of this compile result."""
        return {
            "schema_version": self.request.schema_version,
            "source_config": str(self.config_path),
            "selected_algorithm": self.algorithm,
            "values": deepcopy(dict(self.values)),
            "smoke_test_overrides": deepcopy(dict(self.smoke_test_overrides)),
            "native_options": deepcopy(dict(self.native_options)),
            "robust_config": (
                deepcopy(dict(self.robust_config))
                if self.robust_config is not None else None
            ),
            "request": self.request.snapshot(),
            "task": self.task.to_dict() if self.task is not None else None,
        }
