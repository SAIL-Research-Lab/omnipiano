"""Inference checkpoint shared by the Phase-0 trainers and policy adapter."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path,
    *,
    actor,
    model_spec: dict,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": actor.state_dict(), "model_spec": model_spec},
        path / "model.pt",
    )
    return path
