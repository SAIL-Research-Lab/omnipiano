"""Dependency-light capability catalog for all four experiment tracks.

The catalog is a UI/control-plane view of capabilities that already exist in
the repository.  It never imports MuJoCo, Ray, TorchRL, Stable-Baselines3, or
OmniSafe.  Drift tests compare these declarations with the backend registries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from omnipiano.experiments.models import ExperimentRequest


CATALOG_VERSION = 1
TRACKS = ("standard", "robust", "safety", "marl")


@dataclass(frozen=True)
class AlgorithmCapability:
    name: str
    label: str
    backend: str
    family: str
    status: str = "supported"
    recommended: bool = False
    role: str = "baseline"
    critic: Optional[str] = None
    execution: Optional[str] = None
    required_num_agents: Optional[int] = None
    configurable_parameters: Tuple[str, ...] = ()

    @property
    def selectable(self) -> bool:
        return self.status == "supported"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "backend": self.backend,
            "family": self.family,
            "status": self.status,
            "selectable": self.selectable,
            "recommended": self.recommended,
            "role": self.role,
            "configurable_parameters": list(self.configurable_parameters),
        }
        if self.critic is not None:
            result["critic"] = self.critic
        if self.execution is not None:
            result["execution"] = self.execution
        if self.required_num_agents is not None:
            result["required_num_agents"] = self.required_num_agents
        return result


@dataclass(frozen=True)
class TrackCapability:
    name: str
    label: str
    runtime_profile: str
    current_entrypoint: str
    task_sources: Tuple[str, ...]
    algorithms: Tuple[AlgorithmCapability, ...]
    features: Mapping[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()
    adapter_status: str = "available"

    def algorithm(self, name: str) -> AlgorithmCapability:
        for algorithm in self.algorithms:
            if algorithm.name == name:
                return algorithm
        choices = ", ".join(item.name for item in self.algorithms)
        raise CapabilityError(
            f"algorithm {name!r} is not registered for track {self.name!r}; "
            f"available: {choices}"
        )

    def to_dict(self, *, include_unavailable: bool = False) -> dict[str, Any]:
        algorithms = (
            self.algorithms
            if include_unavailable
            else tuple(item for item in self.algorithms if item.selectable)
        )
        return {
            "name": self.name,
            "label": self.label,
            "runtime_profile": self.runtime_profile,
            "current_entrypoint": self.current_entrypoint,
            "adapter_status": self.adapter_status,
            "task_sources": list(self.task_sources),
            "features": dict(self.features),
            "limitations": list(self.limitations),
            "algorithms": [item.to_dict() for item in algorithms],
        }


class CapabilityError(ValueError):
    """A structurally valid request asks for an unavailable capability."""


def _algo(
    name: str,
    label: str,
    backend: str,
    family: str,
    **kwargs: Any,
) -> AlgorithmCapability:
    return AlgorithmCapability(name, label, backend, family, **kwargs)


_STANDARD_ALGORITHMS = (
    _algo("ppo", "PPO", "stable_baselines3", "on_policy_ppo", recommended=True),
    _algo(
        "sac", "SAC", "stable_baselines3", "off_policy_sac", recommended=True,
        configurable_parameters=("gradient_steps", "learning_starts", "ent_coef"),
    ),
    _algo(
        "td3", "TD3", "stable_baselines3", "off_policy_ddpg",
        configurable_parameters=("gradient_steps", "learning_starts"),
    ),
    _algo(
        "crossq", "CrossQ", "sb3_contrib", "off_policy_sac",
        configurable_parameters=("gradient_steps", "learning_starts"),
    ),
    _algo(
        "tqc", "TQC", "sb3_contrib", "off_policy_sac",
        configurable_parameters=("gradient_steps", "learning_starts", "ent_coef"),
    ),
)


_ROBUST_PPO_PARAMETERS = (
    "net_arch", "n_steps", "n_epochs", "gae_lambda", "clip_epsilon",
    "entropy_coeff", "critic_coeff",
)
_ROBUST_SAC_PARAMETERS = (
    "net_arch", "batch_size", "buffer_size", "learning_starts", "tau",
    "utd", "learning_rate",
)
_ROBUST_ALGORITHMS = (
    _algo(
        "ppo", "PPO", "torchrl", "on_policy_ppo", recommended=True,
        configurable_parameters=_ROBUST_PPO_PARAMETERS,
    ),
    _algo(
        "sac", "SAC", "torchrl", "off_policy_sac", recommended=True,
        configurable_parameters=_ROBUST_SAC_PARAMETERS,
    ),
    _algo(
        "eppo", "EPPO", "torchrl", "on_policy_ppo",
        configurable_parameters=(
            *_ROBUST_PPO_PARAMETERS, "eppo_mode", "kappa", "evidential_reg",
        ),
    ),
    _algo(
        "a2p_sac", "A2P-SAC", "torchrl", "off_policy_sac",
        configurable_parameters=(
            *_ROBUST_SAC_PARAMETERS,
            "a2p_epsilon_mode", "a2p_epsilon", "adversary_learning_rate",
            "adversary_update_freq", "a2p_warmup_steps",
        ),
    ),
    _algo(
        "ompo", "OMPO", "torchrl", "off_policy_sac",
        configurable_parameters=(
            *_ROBUST_SAC_PARAMETERS,
            "ompo_local_buffer_size", "ompo_updates_per_step",
            "ompo_discriminator_observation", "ompo_exponent",
            "ompo_correction_coeff", "ompo_reward_max",
        ),
    ),
    _algo(
        "scpo", "SCPO", "torchrl", "off_policy_sac",
        configurable_parameters=(
            *_ROBUST_SAC_PARAMETERS,
            "scpo_state_noise", "scpo_gradient_observation",
        ),
    ),
)


_SAFETY_ON_POLICY = (
    "PolicyGradient", "NaturalPG", "TRPO", "PPO", "PPOLag", "TRPOLag",
    "RCPO", "PDO", "FOCOPS", "CPO", "PCPO", "OnCRPO", "IPO", "P3O",
    "CUP", "CPPOPID", "TRPOPID", "PPOEarlyTerminated",
    "TRPOEarlyTerminated", "PPOSaute", "TRPOSaute", "PPOSimmerPID",
    "TRPOSimmerPID",
)
_SAFETY_OFF_POLICY = (
    "DDPG", "TD3", "SAC", "DDPGLag", "TD3Lag", "SACLag", "DDPGPID",
    "TD3PID", "SACPID",
)
_SAFETY_PAPER_DEFAULTS = frozenset(("PPO", "PPOLag", "OnCRPO", "CPO", "CUP"))
_SAFETY_ALGORITHMS = tuple(
    _algo(
        name,
        name,
        "omnisafe",
        "on_policy_safe_rl" if name in _SAFETY_ON_POLICY else "off_policy_safe_rl",
        recommended=name in _SAFETY_PAPER_DEFAULTS,
        role="paper_baseline" if name in _SAFETY_PAPER_DEFAULTS else "extended_baseline",
        configurable_parameters=("buffer_size",) if name in _SAFETY_OFF_POLICY else (),
    )
    for name in _SAFETY_ON_POLICY + _SAFETY_OFF_POLICY
)


_MARL_PPO_PARAMETERS = (
    "learning_rate", "critic_learning_rate", "train_batch_size",
    "minibatch_size", "num_epochs", "adam_epsilon", "gae_lambda",
    "clip_param", "value_clip_param", "value_loss_coefficient",
    "entropy_coefficient", "use_kl_loss", "gradient_clip",
    "gradient_clip_mode", "hidden_sizes", "activation", "input_layer_norm",
    "value_norm",
)
_MARL_ALGORITHMS = (
    _algo(
        "ippo", "IPPO", "rllib", "on_policy_ppo", recommended=True,
        critic="own", execution="decentralized",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "ippo-rllib-module", "IPPO · RLlib module control", "rllib",
        "on_policy_ppo", critic="own", execution="decentralized",
        role="diagnostic_control",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "mappo", "MAPPO", "rllib", "on_policy_ppo", recommended=True,
        critic="global", execution="decentralized",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "mappo-own-critic", "MAPPO · own-critic ablation", "rllib",
        "on_policy_ppo", critic="own", execution="decentralized",
        role="ablation",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "happo", "HAPPO", "native", "on_policy_ppo_sequential",
        recommended=True, critic="global", execution="decentralized",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "a2po", "A2PO", "native", "on_policy_ppo_sequential",
        recommended=True, critic="global", execution="decentralized",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "facmac", "FACMAC", "native", "off_policy_ddpg", recommended=True,
        critic="global_joint_action", execution="decentralized",
    ),
    _algo(
        "ppo-monolithic", "PPO · monolithic single-agent baseline", "rllib",
        "on_policy_ppo", critic="own", execution="centralized",
        required_num_agents=1, role="single_agent_control",
        configurable_parameters=_MARL_PPO_PARAMETERS,
    ),
    _algo(
        "mat", "MAT", "native", "on_policy_ppo_sequential", status="planned",
        critic="global", execution="sequential_autoregressive",
    ),
    _algo(
        "masac", "MASAC", "rllib", "off_policy_sac", status="planned",
        critic="global_joint_action", execution="decentralized",
    ),
)


_TRACK_CATALOG = {
    "standard": TrackCapability(
        name="standard",
        label="Standard RL",
        runtime_profile="standard",
        current_entrypoint="examples/run_sb3_baseline.py",
        task_sources=("registered",),
        algorithms=_STANDARD_ALGORITHMS,
        features={
            "evaluation": "periodic_and_final",
            "video": "final_evaluation",
            "tracking": ["csv", "tensorboard"],
        },
        limitations=(
            "The current trainer accepts registered environment IDs only.",
            "Custom MIDI, morphology, and reward compilation await an adapter.",
        ),
    ),
    "robust": TrackCapability(
        name="robust",
        label="Robust RL",
        runtime_profile="robust",
        current_entrypoint="examples/run_torchrl_robust.py",
        task_sources=("registered",),
        algorithms=_ROBUST_ALGORITHMS,
        features={
            "evaluation": "periodic_and_final",
            "video": "final_evaluation",
            "tracking": ["csv"],
            "noise_channels": [
                "action", "observation", "reward", "gravity",
                "contact_friction", "initial_hand_pose",
            ],
            "noise_distributions": ["gaussian", "uniform", "constant_shift"],
        },
        limitations=(
            "Training currently selects an existing registered robust task ID.",
            "Arbitrary channel compositions require an exact registered match.",
        ),
    ),
    "safety": TrackCapability(
        name="safety",
        label="Safe RL",
        runtime_profile="safety",
        current_entrypoint="python -m omnipiano.safety.run",
        task_sources=("registered",),
        algorithms=_SAFETY_ALGORITHMS,
        features={
            "evaluation": "checkpoint_replay",
            "video": "unsupported",
            "tracking": ["tensorboard", "csv"],
            "task_groups": ["main", "hands", "budget", "extensions"],
            "cost_semantics": [
                "joint_range", "actuator_power", "injured_finger",
                "hand_collision",
            ],
            "cost_settings": ["event", "excess", "fraction"],
            "paper_algorithms": sorted(_SAFETY_PAPER_DEFAULTS),
        },
        limitations=(
            "Safety runs require the pinned Python 3.10 OmniSafe runtime.",
            "The zero-intrusion adapter can select existing suite tasks only.",
        ),
    ),
    "marl": TrackCapability(
        name="marl",
        label="Multi-Agent RL",
        runtime_profile="marl",
        current_entrypoint="python -m omnipiano.multiagent.train",
        task_sources=("registered", "custom"),
        algorithms=_MARL_ALGORITHMS,
        features={
            "evaluation": "periodic_and_final",
            "video": "periodic_and_final",
            "tracking": ["wandb", "jsonl"],
            "custom_task": [
                "music", "hands", "agents", "hand_assignment",
                "action_key_ranges", "observation_key_ranges",
                "visible_teammate_hands",
            ],
            "robust_noise_channels": ["observation"],
        },
        limitations=(
            "Action, reward, and physical noise are not wired into MARL.",
            "MAT and MASAC are declared but not launchable.",
        ),
    ),
}


def get_track_capability(track: str) -> TrackCapability:
    try:
        return _TRACK_CATALOG[track]
    except KeyError:
        raise CapabilityError(
            f"unknown track {track!r}; available: {', '.join(TRACKS)}"
        ) from None


def catalog_dict(
    *, track: Optional[str] = None, include_unavailable: bool = False
) -> dict[str, Any]:
    """Return a stable JSON-ready catalog for CLI and the future local API."""
    selected = (
        (get_track_capability(track),)
        if track is not None
        else tuple(_TRACK_CATALOG[name] for name in TRACKS)
    )
    return {
        "catalog_version": CATALOG_VERSION,
        "contract_version": 3,
        "adapter_status": "available",
        "statuses": {
            "supported": "Implemented by the current track backend.",
            "planned": "Declared in the repository but not launchable.",
        },
        "tracks": {
            item.name: item.to_dict(include_unavailable=include_unavailable)
            for item in selected
        },
    }


def validate_request_capabilities(request: ExperimentRequest) -> None:
    """Reject valid JSON that asks a track to do something it cannot do."""
    track = get_track_capability(request.design.type)
    if request.task.source not in track.task_sources:
        allowed = ", ".join(track.task_sources)
        raise CapabilityError(
            f"track {track.name!r} does not support task source "
            f"{request.task.source!r}; supported: {allowed}"
        )
    algorithm = track.algorithm(request.algorithm.name)
    if not algorithm.selectable:
        raise CapabilityError(
            f"algorithm {algorithm.name!r} is {algorithm.status} and cannot be "
            "selected for a runnable experiment"
        )
