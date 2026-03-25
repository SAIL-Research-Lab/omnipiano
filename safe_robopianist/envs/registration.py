"""Environment registration and factory for SafeRoboPianist.

This module implements the register() + make() pattern (following Robust-Gymnasium).
- register(): Declares a benchmark task by storing its TaskSpec in the registry.
- make(): Looks up the registry, resolves configs, and builds the full wrapper chain.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import gymnasium as gym
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from robopianist import suite
from robopianist.wrappers.evaluation import MidiEvaluationWrapper

from safe_robopianist.wrappers.robust_wrapper import RobustWrapper
from safe_robopianist.wrappers.safety_wrapper import SafetyWrapper
from safe_robopianist.wrappers.metrics_wrapper import MetricsWrapper
from safe_robopianist.configs import RobustConfig, SafetyConfig, TaskVariantConfig
from safe_robopianist.tasks.safe_piano_task import SafePianoTask


@dataclass
class TaskSpec:
    """Specification for a registered benchmark task."""
    base_env_name: str
    safety_config: Optional[SafetyConfig] = None
    robust_config: Optional[RobustConfig] = None
    task_config: Optional[TaskVariantConfig] = None


_registry: Dict[str, TaskSpec] = {}


def register(id: str, **kwargs):
    """Register a benchmark task by name.

    Args:
        id: Unique task identifier, e.g. "SafeRoboPianist-Twinkle-RightHandWristLimit-v0".
        **kwargs: Fields of TaskSpec (base_env_name, safety_config, robust_config, task_config).
    """
    _registry[id] = TaskSpec(**kwargs)


def make(
    env_name: str,
    robust_config: RobustConfig = None,
    safety_config: SafetyConfig = None,
    task_config: TaskVariantConfig = None,
    log_dir: str = None,
    log_split: str = "train",
    record_dir: str = None,
    **kwargs
):
    """Factory function to create a Safe/Robust RoboPianist environment.

    If env_name is a registered task, its default configs are used unless
    explicitly overridden by the caller.
    """
    # 0. Look up registry for pre-defined tasks
    if env_name in _registry:
        task_spec = _registry[env_name]
        base_env_name = task_spec.base_env_name

        if robust_config is None and task_spec.robust_config is not None:
            robust_config = task_spec.robust_config
        if safety_config is None and task_spec.safety_config is not None:
            safety_config = task_spec.safety_config
        if task_config is None and task_spec.task_config is not None:
            task_config = task_spec.task_config
    else:
        base_env_name = env_name

    if robust_config is None:
        robust_config = RobustConfig()
    if safety_config is None:
        safety_config = SafetyConfig()
    if task_config is None:
        task_config = TaskVariantConfig()

    # 1. Extract environment parameters from kwargs
    midi_file_path = kwargs.pop("midi_file", None)
    seed = kwargs.pop("seed", None)

    # 2. Compile into a dm_env
    task_kwargs = {"task_config": task_config, **kwargs}

    dm_env = suite.load_with_task(
        environment_name=base_env_name,
        task_cls=SafePianoTask,
        midi_file=midi_file_path,
        seed=seed,
        task_kwargs=task_kwargs
    )

    # 3. Add Musical Evaluation Wrapper (dm_env level)
    dm_env = MidiEvaluationWrapper(dm_env, deque_size=1)

    if record_dir is not None:
        from robopianist.wrappers.sound import PianoSoundVideoWrapper
        dm_env = PianoSoundVideoWrapper(
            dm_env,
            record_dir=record_dir,
            record_every=1,
            camera_id="piano/back",
        )

    # 4. Convert dm_env to Gymnasium environment
    # Pipeline: temporarily monkey-patch shimmy's env-type detector so that
    # our custom dm_env wrappers are recognized as "dm_control", then restore.
    import shimmy.dm_control_compatibility
    original_find_env_type = (
        shimmy.dm_control_compatibility.DmControlCompatibilityV0._find_env_type
    )

    def custom_find_env_type(self, env):
        env_classes = [MidiEvaluationWrapper]
        try:
            from robopianist.wrappers.sound import PianoSoundVideoWrapper
            env_classes.append(PianoSoundVideoWrapper)
        except ImportError:
            pass

        if isinstance(env, tuple(env_classes)):
            return "dm_control"
        return original_find_env_type(self, env)

    shimmy.dm_control_compatibility.DmControlCompatibilityV0._find_env_type = (
        custom_find_env_type
    )
    try:
        gym_env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
    finally:
        shimmy.dm_control_compatibility.DmControlCompatibilityV0._find_env_type = (
            original_find_env_type
        )

    # 5. Normalize action space to [-1, 1]
    gym_env = gym.wrappers.RescaleAction(gym_env, min_action=-1.0, max_action=1.0)

    # 6. Apply Modular Wrappers
    # Metrics (innermost) -> Safety (middle) -> Robust (outermost)
    env = MetricsWrapper(gym_env)
    env = SafetyWrapper(env, config=safety_config)
    env = RobustWrapper(env, config=robust_config)

    # 7. Add episode CSV logger only for eval split.
    if log_dir is not None and log_split == "eval":
        from safe_robopianist.utils.logger_wrapper import SafeRecordEpisodeStatistics
        env = SafeRecordEpisodeStatistics(
            env,
            log_dir=log_dir,
            split=log_split,
        )

    return env
