"""Backward-compatible imports for the relocated shared PPO learner.

The canonical implementation lives in :mod:`omnipiano.multiagent.algos.ppo_learner`.
This shim preserves historical imports and checkpoint class references.
"""

from omnipiano.multiagent.algos.ppo_learner import (
    GAEWithOldValuePredictions,
    OLD_NORMALIZED_VALUE_PREDS,
    OmniPianoPPOTorchLearner,
    VALUE_CLIP_FRACTION_KEY,
    VALUE_NORM_MEAN_KEY,
    VALUE_NORM_STD_KEY,
)

__all__ = [
    "GAEWithOldValuePredictions",
    "OLD_NORMALIZED_VALUE_PREDS",
    "OmniPianoPPOTorchLearner",
    "VALUE_CLIP_FRACTION_KEY",
    "VALUE_NORM_MEAN_KEY",
    "VALUE_NORM_STD_KEY",
]
