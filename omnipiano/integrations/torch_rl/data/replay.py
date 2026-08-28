"""TorchRL replay-buffer factories."""

from __future__ import annotations


def make_replay_buffer(capacity: int, batch_size: int, scratch_dir=None):
    from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer

    storage = (
        LazyMemmapStorage(capacity, scratch_dir=scratch_dir)
        if scratch_dir is not None
        else LazyTensorStorage(capacity)
    )
    return TensorDictReplayBuffer(storage=storage, batch_size=batch_size)
