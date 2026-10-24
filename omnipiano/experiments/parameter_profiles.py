"""JSON-compatible editable defaults exposed by the unified experiment UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_PPO_SB3 = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "stats_window_size": 100,
    "net_arch": [64, 64],
}

_OFF_POLICY_COMMON = {
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "train_freq": 1,
    "gradient_steps": -1,
    "optimize_memory_usage": False,
    "n_steps": 1,
    "stats_window_size": 100,
}

_SAC_COMMON = {
    **_OFF_POLICY_COMMON,
    "learning_rate": 3e-4,
    "tau": 0.005,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "net_arch": [256, 256],
}

STANDARD_PARAMETER_TEMPLATES: Mapping[str, Mapping[str, Any]] = {
    "ppo": _PPO_SB3,
    "sac": {
        **_SAC_COMMON,
        "target_update_interval": 1,
    },
    "td3": {
        **_OFF_POLICY_COMMON,
        "learning_rate": 1e-3,
        "tau": 0.005,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "net_arch": [400, 300],
    },
    "crossq": {
        **_OFF_POLICY_COMMON,
        "learning_rate": 1e-3,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "policy_delay": 3,
        "use_sde": False,
        "sde_sample_freq": -1,
        "use_sde_at_warmup": False,
        "net_arch": [256, 256],
    },
    "tqc": {
        **_SAC_COMMON,
        "target_update_interval": 1,
        "top_quantiles_to_drop_per_net": 2,
    },
}

_ROBUST_PPO = {
    "net_arch": [256, 256],
    "n_steps": 2048,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coeff": 0.0,
    "critic_coeff": 0.5,
}
_ROBUST_SAC = {
    "net_arch": [256, 256],
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "learning_starts": 5_000,
    "tau": 0.005,
    "utd": 1,
    "learning_rate": 3e-4,
}
ROBUST_PARAMETER_TEMPLATES: Mapping[str, Mapping[str, Any]] = {
    "ppo": _ROBUST_PPO,
    "sac": _ROBUST_SAC,
    "eppo": {
        **_ROBUST_PPO,
        "eppo_mode": "cor",
        "kappa": 0.05,
        "evidential_reg": 0.001,
    },
    "a2p_sac": {
        **_ROBUST_SAC,
        "a2p_epsilon_mode": "adaptive",
        "a2p_epsilon": 0.1,
        "adversary_learning_rate": 3e-4,
        "adversary_update_freq": 10,
        "a2p_warmup_steps": None,
    },
    "ompo": {
        **_ROBUST_SAC,
        "ompo_local_buffer_size": 1_000,
        "ompo_updates_per_step": 3,
        "ompo_discriminator_observation": "dynamic",
        "ompo_exponent": 1.5,
        "ompo_correction_coeff": 0.001,
        "ompo_reward_max": 1.0,
    },
    "scpo": {
        **_ROBUST_SAC,
        "scpo_state_noise": 0.005,
        "scpo_gradient_observation": "dynamic",
    },
}

_MARL_PPO = {
    "learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "train_batch_size": 4_000,
    "minibatch_size": 4_000,
    "num_epochs": 5,
    "adam_epsilon": 1e-5,
    "gae_lambda": 0.95,
    "clip_param": 0.2,
    "value_clip_param": 0.2,
    "value_loss_coefficient": 1.0,
    "entropy_coefficient": 0.0,
    "use_kl_loss": False,
    "gradient_clip": 10.0,
    "gradient_clip_mode": "global_norm",
    "hidden_sizes": [256, 256],
    "activation": "tanh",
    "hidden_orthogonal_gain": 2 ** 0.5,
    "policy_output_gain": 0.01,
    "value_output_gain": 1.0,
    "initial_log_std": 0.0,
    "log_std_min": -5.0,
    "log_std_max": 2.0,
    "input_layer_norm": True,
    "value_norm": True,
    "value_norm_beta": 0.99999,
    "value_norm_epsilon": 1e-5,
    "value_norm_variance_floor": 0.01,
}
_A2PO_NATIVE = {
    "order_score_epsilon": 1e-8,
    "preceding_ratio_clip": 0.1,
    "adaptive_clip_weight": 0.5,
}
_FACMAC_NATIVE = {
    "actor_hidden_sizes": [400, 400],
    "utility_hidden_sizes": [400, 400],
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "adam_epsilon": 0.01,
    "grad_clip": 0.5,
    "replay_capacity": 1_000_000,
    "batch_size": 100,
    "buffer_warmup": 1_000,
    "random_action_steps": 10_000,
    "collect_steps": 24,
    "updates_per_env_step": 1.0,
    "tau": 0.001,
    "noise_std": 0.1,
    "action_l2": 0.001,
    "mixer_embed": 64,
    "hypernet_embed": 64,
    "max_replay_gib": 32.0,
    "checkpoint_replay_transitions": 50_000,
}
MARL_PARAMETER_TEMPLATES: Mapping[str, Mapping[str, Any]] = {
    "ippo": _MARL_PPO,
    "ippo-rllib-module": _MARL_PPO,
    "mappo": _MARL_PPO,
    "mappo-own-critic": _MARL_PPO,
    "happo": _MARL_PPO,
    "a2po": {**_MARL_PPO, **_A2PO_NATIVE},
    "facmac": _FACMAC_NATIVE,
    "ppo-monolithic": _MARL_PPO,
}


def copy_template(profiles: Mapping[str, Mapping[str, Any]], algorithm: str) -> dict[str, Any]:
    return deepcopy(dict(profiles[algorithm]))
