"""Thin OmniPiano Gymnasium-to-TorchRL adapter."""

from __future__ import annotations

from functools import partial
from typing import Literal


def make_env(
    env_id: str,
    seed: int,
    mode: Literal["train", "eval"],
    eval_noise_scale: float | None = None,
    log_dir: str | None = None,
):
    """Build one seeded :class:`torchrl.envs.EnvBase`.

    Evaluation keeps OmniPiano's native CSV wrapper when ``log_dir`` is set.
    Training deliberately does not attach that wrapper.
    """
    if mode not in ("train", "eval"):
        raise ValueError("mode must be 'train' or 'eval'")

    import omnipiano
    from torchrl.envs import GymWrapper
    from omnipiano.utils.info_keys import InfoKeys

    kwargs = {"seed": seed, "mode": mode, "log_dir": log_dir}
    if mode == "eval":
        kwargs["eval_noise_scale"] = (
            1.0 if eval_noise_scale is None else eval_noise_scale
        )
    gym_env = omnipiano.make(env_id, **kwargs)
    return GymWrapper(gym_env, info_keys=[InfoKeys.TASK_TRUE_REWARD],)


def make_parallel_env(env_id: str, seed: int, n_envs: int):
    """Create the training environment used by a TorchRL collector."""
    if n_envs == 1:
        return make_env(env_id, seed, "train")

    from torchrl.envs import ParallelEnv

    makers = [
        partial(make_env, env_id, seed + worker_index, "train")
        for worker_index in range(n_envs)
    ]
    return ParallelEnv(n_envs, makers)


def make_eval_env(env_id: str, seed: int, log_dir: str, record_dir: str | None = None):
    """Create the native Gymnasium environment used for benchmark evaluation."""
    import omnipiano

    kwargs = {
        "mode": "eval",
        "log_dir": log_dir,
        "seed": seed,
    }
    if record_dir is not None:
        kwargs["record_dir"] = record_dir
    return omnipiano.make(env_id, **kwargs)
