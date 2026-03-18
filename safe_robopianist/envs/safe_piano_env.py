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
from safe_robopianist.tasks.registry import REGISTERED_TASKS

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
    # 0. Check Registry for pre-defined tasks
    if env_name in REGISTERED_TASKS:
        task_spec = REGISTERED_TASKS[env_name]
        base_env_name = task_spec.base_env_name
        
        # Override configs if they are provided in the registry AND not explicitly passed
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
    
    # 2. Compile into a dm_env using the new custom load function
    task_kwargs = {"task_config": task_config, **kwargs}
    
    dm_env = suite.load_with_task(
        environment_name=base_env_name,
        task_cls=SafePianoTask,
        midi_file=midi_file_path,
        seed=seed,
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
    # TODO(CRITICAL): Re-evaluate Wrapper ordering when implementing specific tasks.
    # Current hypothesis:
    # - Metrics (innermost): Needs to read raw, unpenalized reward from physics.
    # - Safety (middle): Needs to evaluate the *actual* action (including noise) executed by physics.
    # - Robust (outermost): Needs to inject noise into action before safety checks it, 
    #   and inject noise into obs before agent sees it.
    # This order is subject to change based on future safety/robustness task definitions.
    
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