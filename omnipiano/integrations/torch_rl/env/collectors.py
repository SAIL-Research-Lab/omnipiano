"""Collector construction and environment-step accounting."""

from __future__ import annotations

def make_collector(env, policy, frames_per_batch, total_frames, device: str):
    from torchrl.collectors import Collector

    return Collector(
        create_env_fn=env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        policy_device=device,
        reset_when_done=True,
        auto_register_policy_transforms=True,
    )


def batch_env_steps(batch) -> int:
    """TorchRL's batch numel is the vec-env-aggregated interaction count."""
    return int(batch.numel())
