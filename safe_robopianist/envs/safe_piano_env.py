import gymnasium as gym
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from robopianist import suite
from robopianist.music import midi_file
from robopianist import music
from robopianist.suite import composer_utils
from robopianist.wrappers.evaluation import MidiEvaluationWrapper

from safe_robopianist.wrappers.robust_wrapper import RobustWrapper
from safe_robopianist.wrappers.safety_wrapper import SafetyWrapper
from safe_robopianist.wrappers.metrics_wrapper import MetricsWrapper
from safe_robopianist.configs import RobustConfig, SafetyConfig, TaskVariantConfig
from safe_robopianist.tasks.safe_piano_task import SafePianoTask

def make(
    env_name: str,
    robust_config: RobustConfig = None,
    safety_config: SafetyConfig = None,
    task_config: TaskVariantConfig = None,
    log_dir: str = None,
    log_split: str = "train",
    **kwargs
):
    """
    Factory function to create a Safe/Robust RoboPianist environment.
    """
    if robust_config is None:
        robust_config = RobustConfig()
    if safety_config is None:
        safety_config = SafetyConfig()
    if task_config is None:
        task_config = TaskVariantConfig()

    # 1. Extract environment parameters from kwargs
    midi_file_path = kwargs.pop("midi_file", None)
    stretch = kwargs.pop("stretch", 1.0)
    shift = kwargs.pop("shift", 0)
    seed = kwargs.pop("seed", None)
    recompile_physics = kwargs.pop("recompile_physics", False)
    legacy_step = kwargs.pop("legacy_step", True)
    
    # 2. Compile into a dm_env using the new custom load function
    task_kwargs = {"task_config": task_config, **kwargs}
    
    dm_env = suite.load_with_task(
        environment_name=env_name,
        task_cls=SafePianoTask,
        midi_file=midi_file_path,
        seed=seed,
        stretch=stretch,
        shift=shift,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
        task_kwargs=task_kwargs
    )
    
    # 3.5 Add Musical Evaluation Wrapper (dm_env level)
    dm_env = MidiEvaluationWrapper(dm_env, deque_size=1)
    
    # 2. Convert dm_env to Gymnasium environment
    import shimmy.dm_control_compatibility
    original_find_env_type = (
        shimmy.dm_control_compatibility.DmControlCompatibilityV0._find_env_type
    )

    def custom_find_env_type(self, env):
        if isinstance(env, MidiEvaluationWrapper):
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

    # 3. Normalize action space to [-1, 1]
    gym_env = gym.wrappers.RescaleAction(gym_env, min_action=-1.0, max_action=1.0)
    
    # 4. Apply Modular Wrappers
    # Order matters: Metrics (innermost) -> Safety -> Robust (outermost)
    # This ensures safety costs are calculated on the noisy actions, 
    # and metrics gather all info before returning.
    
    env = MetricsWrapper(gym_env)
    env = SafetyWrapper(env, config=safety_config)
    env = RobustWrapper(env, config=robust_config)
    
    # 5. Add episode CSV logger only for eval split.
    if log_dir is not None and log_split == "eval":
        from safe_robopianist.logging.logger_wrapper import SafeRecordEpisodeStatistics
        env = SafeRecordEpisodeStatistics(
            env,
            log_dir=log_dir,
            split=log_split,
        )
    
    return env
